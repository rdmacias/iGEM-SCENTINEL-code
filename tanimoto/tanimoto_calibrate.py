# Grid-search calibration for the Tanimoto-Ki read-across model.
# For every combination of (t_min, k, p), runs Leave-One-Out validation
# (via tanimoto_impute.run_loo) and reports RMSE, keeping the best
# combination to recommend for the final imputation step
# (generate_imputed_csvs.py). Also runs a permutation-based significance
# test on the winning combination's Spearman correlation, to check that the
# prediction skill isn't just noise.

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tanimoto_impute as ti

BINDING_FILE = "Compound_OBP_binding.csv"

# Parameter grid to try (extend if needed)
T_MIN_VALUES = [0.30, 0.35, 0.45, 0.55]  # minimum Tanimoto similarity to accept a donor
K_VALUES     = [3, 5, 8]                  # number of nearest neighbours (donors) used
P_VALUES     = [1.0, 2.0]                 # exponent for similarity-based weighting

# Minimum fraction of known cells a combination must be able to predict (via
# LOO) to be eligible as the "best" one. Without this floor, plain RMSE
# minimization tends to pick a high t_min that fits its (small) covered
# subset slightly better while leaving most of the matrix unfillable.
MIN_COVERAGE = 0.60

os.makedirs("results", exist_ok=True)


def significance_test(df_err, n_permutations=1000, seed=42):
    """
    Permutation-based significance test for the Spearman correlation between
    predicted and real pKi. Shuffles the predictions many times to build a
    null distribution of correlations expected by chance, then compares the
    observed correlation against it.

    Returns:
        - rho_real: observed correlation
        - p_value: p-value based on permutations
        - rho_mean_random: mean of the random (null) correlations
        - rho_std_random: standard deviation of the random correlations
        - rho_distribution: array with all random correlations
        - confidence_interval: 95% confidence interval for random correlations
    """
    np.random.seed(seed)

    # Observed (real) correlation
    rho_real, pval_real = spearmanr(df_err["pred_pki"], df_err["real_pki"])

    # Bootstrap/permutation: shuffle predictions to break any real relationship
    random_rhos = []
    for _ in range(n_permutations):
        shuffled_pred = np.random.permutation(df_err["pred_pki"].values)
        r, _ = spearmanr(df_err["real_pki"], shuffled_pred)
        random_rhos.append(r)

    random_rhos = np.array(random_rhos)

    # Summary statistics of the null (random) distribution
    rho_mean_random = np.mean(random_rhos)
    rho_std_random = np.std(random_rhos)

    # P-value = proportion of random correlations at least as extreme as the
    # observed one (one-sided, direction depends on the sign of rho_real)
    if rho_real >= 0:
        p_value = np.mean(random_rhos >= rho_real)
    else:
        p_value = np.mean(random_rhos <= rho_real)

    # 95% confidence interval for correlations expected by chance alone
    lower_percentile = 2.5
    upper_percentile = 97.5
    ci_lower = np.percentile(random_rhos, lower_percentile)
    ci_upper = np.percentile(random_rhos, upper_percentile)

    return {
        'rho_real': rho_real,
        'pval_parametric': pval_real,  # p-value from scipy
        'pval_bootstrap': p_value,
        'rho_mean_random': rho_mean_random,
        'rho_std_random': rho_std_random,
        'rho_distribution': random_rhos,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'n_permutations': n_permutations,
        'n_predictions': len(df_err)
    }


def plot_significance_test(sig_results, output_path="results/significance_test.png"):
    """
    Plot the distribution of random (null) correlations alongside the
    observed real correlation, plus a scatter of predicted vs. real pKi.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: histogram of random correlations
    ax1 = axes[0]
    random_rhos = sig_results['rho_distribution']
    rho_real = sig_results['rho_real']

    # Histogram of the null distribution
    n, bins, patches = ax1.hist(random_rhos, bins=50, alpha=0.7,
                                color='#94a3b8', edgecolor='black', linewidth=0.5,
                                label=f'Random distribution\n(n={sig_results["n_permutations"]})')

    # Vertical line marking the observed (real) correlation
    ax1.axvline(x=rho_real, color='#2563eb', linewidth=2.5,
                label=f'observed rho = {rho_real:.3f}')

    # 95% confidence interval for chance correlations
    ax1.axvline(x=sig_results['ci_95_lower'], color='red', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'random 95% CI: [{sig_results["ci_95_lower"]:.3f}, {sig_results["ci_95_upper"]:.3f}]')
    ax1.axvline(x=sig_results['ci_95_upper'], color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)

    # Shade the 95% CI region
    ax1.axvspan(sig_results['ci_95_lower'], sig_results['ci_95_upper'],
                alpha=0.15, color='red', label='random 95% CI')

    ax1.set_xlabel('Spearman rho (correlation)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f"Spearman significance test\n"
                  f"p-value = {sig_results['pval_bootstrap']:.4f} "
                  f"({sig_results['n_predictions']} predictions)", fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: scatter plot with annotated statistics
    ax2 = axes[1]
    df_err = sig_results.get('df_err', None)
    if df_err is not None:
        ax2.scatter(df_err["real_pki"], df_err["pred_pki"],
                   alpha=0.4, s=25, color='#2563eb', edgecolors='white', linewidth=0.5)

        lims = [
            min(df_err["real_pki"].min(), df_err["pred_pki"].min()),
            max(df_err["real_pki"].max(), df_err["pred_pki"].max()),
        ]
        ax2.plot(lims, lims, color="gray", linestyle="--", linewidth=1.5,
                label="Perfect prediction")

        # Annotate the statistics
        stats_text = (f"Spearman rho = {sig_results['rho_real']:.3f}\n"
                     f"p-value = {sig_results['pval_bootstrap']:.4f}\n"
                     f"n = {len(df_err)}")
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_xlabel("Real pKi (experimental)", fontsize=12)
        ax2.set_ylabel("Predicted pKi (Tanimoto read-across)", fontsize=12)
        ax2.set_title("Predicted vs Real (LOO)", fontsize=13)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  OK Significance plot saved to {output_path}")


def interpret_significance(sig_results):
    """
    Print a human-readable interpretation of the significance test results.
    """
    print("\n" + "=" * 70)
    print("  SPEARMAN SIGNIFICANCE TEST")
    print("=" * 70)

    rho = sig_results['rho_real']
    pval = sig_results['pval_bootstrap']
    n = sig_results['n_predictions']

    print(f"\n  Observed Spearman correlation: rho = {rho:.4f}")
    print(f"  P-value (bootstrap, n_perm={sig_results['n_permutations']}): {pval:.4f}")
    print(f"  Number of predictions: {n}")

    # Null (random) distribution
    print(f"\n  Distribution by chance (mean): {sig_results['rho_mean_random']:.4f} +/- {sig_results['rho_std_random']:.4f}")
    print(f"  95% CI by chance: [{sig_results['ci_95_lower']:.4f}, {sig_results['ci_95_upper']:.4f}]")

    # Interpretation
    print("\n  INTERPRETATION:")

    if rho > 0.5 and pval < 0.05:
        print("  [OK] STRONGLY SIGNIFICANT CORRELATION")
        print("     - The correlation is NOT due to chance (p < 0.05)")
        print("     - Correlation strength: STRONG (rho > 0.5)")
        if rho > 0.7:
            print("     - The prediction is very robust")
    elif rho > 0.3 and pval < 0.05:
        print("  [OK] SIGNIFICANT CORRELATION")
        print("     - The correlation is NOT due to chance (p < 0.05)")
        print("     - Correlation strength: MODERATE (0.3 < rho < 0.5)")
    elif pval < 0.05:
        print("  [!] WEAK BUT SIGNIFICANT CORRELATION")
        print("     - The correlation is NOT due to chance (p < 0.05)")
        print("     - But the strength is weak (rho < 0.3)")
    else:
        print("  [X] NON-SIGNIFICANT CORRELATION")
        print(f"     - The correlation COULD BE DUE TO CHANCE (p = {pval:.4f} > 0.05)")
        print("     - Check: sample size, data quality, or parameters")

    # Comparison against the parametric p-value
    print(f"\n  P-value comparison:")
    print(f"    - Parametric (scipy): {sig_results['pval_parametric']:.4e}")
    print(f"    - Bootstrap:          {sig_results['pval_bootstrap']:.4f}")

    # Advice
    print("\n  RECOMMENDATIONS:")
    if n < 30:
        print("    [!] Small sample (< 30) - the p-values may be unstable")
    if abs(rho - sig_results['rho_mean_random']) < 2 * sig_results['rho_std_random']:
        print("    [!] The real correlation lies within 2 sigma of the random distribution")
        print("       -> It could be a false positive")
    elif pval < 0.01:
        print("    [OK] Very robust result (p < 0.01)")


def main():
    print("=" * 70)
    print("  TANIMOTO-Ki CALIBRATION (Leave-One-Out)")
    print("=" * 70)

    print(f"\n  Reading {BINDING_FILE}...")
    df = pd.read_csv(BINDING_FILE)
    cas_col = df.columns[0]
    obp_name_list = list(df.columns[2:])

    # Convert Ki values the same way main.py does (censored values like
    # ">X" are treated as X*1.1, a conservative estimate of the true Ki)
    def convert_ki(raw):
        if pd.isna(raw):
            return np.nan
        text = str(raw).strip().replace('\xa0', '').replace(' ', '')
        if text.startswith('>'):
            try:
                import re
                return float(re.sub(r'[^\d.]', '', text)) * 1.1
            except ValueError:
                return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan

    binding_table = df.copy()
    for col in obp_name_list:
        binding_table[col] = binding_table[col].apply(convert_ki)

    n_known = int(binding_table[obp_name_list].notna().sum().sum())
    print(f"  -> {n_known} cells with a known Ki (they will be used to validate)")

    print(f"\n  Loading the Tanimoto matrix...")
    T_matrix, cas_to_pos = ti.load_tanimoto_data()
    idx_to_pos = ti.build_binding_idx_to_matrix_pos(binding_table, cas_col, cas_to_pos)
    print(f"  -> {len(idx_to_pos)}/{len(binding_table)} VOCs with a valid position in the matrix")

    # Grid search: run LOO validation for every (t_min, k, p) combination and
    # keep track of all of them so we can pick the winner by RMSE *among
    # those that clear the coverage floor* (see MIN_COVERAGE above).
    results = []
    df_errs_by_combo = {}

    total_combos = len(T_MIN_VALUES) * len(K_VALUES) * len(P_VALUES)
    combo_i = 0

    for t_min in T_MIN_VALUES:
        for k in K_VALUES:
            for p in P_VALUES:
                combo_i += 1
                print(f"\n  [{combo_i}/{total_combos}] Trying t_min={t_min}  k={k}  p={p}")
                rmse, df_err = ti.run_loo(
                    binding_table, T_matrix, idx_to_pos, obp_name_list,
                    t_min=t_min, k=k, p=p,
                )
                if rmse is None:
                    print(f"     No valid predictions with this combination")
                    continue

                n_pred = len(df_err)
                coverage = n_pred / n_known
                print(f"    -> RMSE={rmse:.3f}  (n={n_pred}, coverage={coverage:.1%})")

                combo_key = (t_min, k, p)
                results.append({
                    "t_min": t_min, "k": k, "p": p,
                    "RMSE": round(rmse, 4),
                    "n_predicted": n_pred,
                    "coverage": round(coverage, 4),
                })
                df_errs_by_combo[combo_key] = df_err

    if not results:
        print("\n   No combination produced results. Check the Tanimoto matrix.")
        return

    # Summary of all combinations tried, sorted by RMSE
    df_summary = pd.DataFrame(results).sort_values("RMSE")
    out1 = "results/loo_summary_combinations.csv"
    df_summary.round(2).to_csv(out1, index=False)

    print("\n" + "=" * 70)
    print("  SUMMARY (sorted by RMSE, lower = better)")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(f"\n   Full summary saved to {out1}")

    # Pick the winner: among combos with coverage >= MIN_COVERAGE, take the
    # lowest RMSE. If none clear that floor, fall back to the combo with the
    # highest coverage available (RMSE as tiebreaker) rather than silently
    # returning a barely-covered "best RMSE" combination.
    qualifying = [r for r in results if r["coverage"] >= MIN_COVERAGE]
    if qualifying:
        best_r = min(qualifying, key=lambda r: r["RMSE"])
    else:
        print(f"\n   No combination reaches MIN_COVERAGE={MIN_COVERAGE:.0%}; "
              f"falling back to the one with the highest available coverage.")
        best_r = max(results, key=lambda r: (r["coverage"], -r["RMSE"]))

    combo_key = (best_r["t_min"], best_r["k"], best_r["p"])
    best = {
        "rmse": best_r["RMSE"],
        "t_min": best_r["t_min"],
        "k": best_r["k"],
        "p": best_r["p"],
        "df_err": df_errs_by_combo[combo_key],
    }

    #  Detail of the winning combination
    print("\n" + "=" * 70)
    print(f"  BEST COMBINATION: t_min={best['t_min']}  k={best['k']}  "
          f"p={best['p']}  (RMSE={best['rmse']:.3f})")
    print("=" * 70)

    df_err = best["df_err"]
    out2 = "results/loo_errors_detail.csv"
    df_err.round(2).to_csv(out2, index=False)
    print(f"  OK Individual errors saved to {out2}")

    summary_bins = ti.rmse_by_tmax_bin(df_err)
    out3 = "results/loo_rmse_per_tmax.csv"
    summary_bins.round(2).to_csv(out3)
    print(f"\n  RMSE broken down by the similarity of the best neighbour (T_max):")
    print(summary_bins.to_string())
    print(f"\n  OK Saved to {out3}")

    #  SPEARMAN SIGNIFICANCE TEST (via bootstrapping/permutation)
    print("\n" + "=" * 70)
    print("  COMPUTING THE SPEARMAN SIGNIFICANCE TEST...")
    print("=" * 70)

    sig_results = significance_test(df_err, n_permutations=5000, seed=42)
    sig_results['df_err'] = df_err  # keep the data around for plotting

    # Save results to CSV
    sig_summary = {
        'rho_real': sig_results['rho_real'],
        'pval_parametric': sig_results['pval_parametric'],
        'pval_bootstrap': sig_results['pval_bootstrap'],
        'rho_mean_random': sig_results['rho_mean_random'],
        'rho_std_random': sig_results['rho_std_random'],
        'ci_95_lower': sig_results['ci_95_lower'],
        'ci_95_upper': sig_results['ci_95_upper'],
        'n_predictions': sig_results['n_predictions'],
        'n_permutations': sig_results['n_permutations']
    }
    pd.DataFrame([sig_summary]).round(2).to_csv("results/spearman_significance.csv", index=False)
    print(f"  OK Significance summary saved to results/spearman_significance.csv")

    # Print interpretation
    interpret_significance(sig_results)

    # Significance plot
    plot_significance_test(sig_results, output_path="results/spearman_significance_test.png")

    #  Original scatter plot (prediction vs real + Spearman rho)
    rho = sig_results['rho_real']
    pval = sig_results['pval_parametric']

    plt.figure(figsize=(7, 7))
    plt.scatter(df_err["real_pki"], df_err["pred_pki"], alpha=0.3, s=10, color="#2563eb")

    lims = [
        min(df_err["real_pki"].min(), df_err["pred_pki"].min()),
        max(df_err["real_pki"].max(), df_err["pred_pki"].max()),
    ]
    plt.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Perfect prediction")

    plt.xlabel("Real pKi (experimental)")
    plt.ylabel("Predicted pKi (Tanimoto read-across)")

    # Add the significance information to the title
    title = (f"Predicted vs Real (LOO)\n"
             f"Spearman rho = {rho:.3f}  (p = {pval:.2e})\n"
             f"RMSE = {best['rmse']:.3f}  n = {len(df_err)}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out4 = "results/spearman_scatter.png"
    plt.savefig(out4, dpi=150)
    print(f"  OK Plot saved to {out4}")

    print("\n" + "=" * 70)
    print("  RMSE INTERPRETATION")
    print("=" * 70)
    print("  RMSE <= 0.3 (log10 Ki)  ->  error factor <= ~2x   -> GOOD")
    print("  RMSE 0.3-0.5            ->  error factor ~2-3x    -> acceptable")
    print("  RMSE > 0.5              ->  error factor > 3x     -> too noisy")
    print(f"\n  -> Recommended parameters for the next step (generate_imputed_csvs.py):")
    print(f"      T_MIN = {best['t_min']}")
    print(f"      K     = {best['k']}")
    print(f"      P     = {best['p']}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Number of predictions: {len(df_err)}")
    print(f"  RMSE: {best['rmse']:.4f}")
    print(f"  Spearman rho: {rho:.4f}")
    print(f"  P-value (bootstrap): {sig_results['pval_bootstrap']:.4f}")
    print(f"  Coverage: {len(df_err)/n_known:.2%}")
    if sig_results['pval_bootstrap'] < 0.05:
        print("\n  [OK] CONCLUSION: the correlation is SIGNIFICANT (not due to chance)")
    else:
        print("\n  [!] CONCLUSION: the correlation COULD BE DUE TO CHANCE (p >= 0.05)")


if __name__ == "__main__":
    main()
