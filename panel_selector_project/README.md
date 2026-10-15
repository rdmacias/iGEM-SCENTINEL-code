# panel_selector

Mòdul Python nou i independent que implementa les dues eines descrites a
`IDEA_SOFTWARE.pdf`, construïdes sobre el mateix nucli compartit (matriu de
sensibilitat + model directe + optimitzador greedy). No toca ni depèn del
codi de SCENTINEL v3, però llegeix el mateix tipus de taula de Ki
(OBP × compost, en µM) si en el futur vols apuntar-lo a dades reals.

## Idea general

1. **Matriu de sensibilitat S** (OBPs × compostos): `s_ji ≈ 1/Ki`.
2. **Model directe** (règim lineal): `resposta_j = Σ_i s_ji · c_i`.
3. Amb això, dues eines — mateix bucle greedy, funció objectiu diferent:
   - **Eina 1 (`specificity.py`)**: panel mínim tal que cap combinació
     no-negativa d'interferents pugui imitar la diana (MeSA, o DIPP/DMP en
     una barreja). Es resol amb NNLS (`scipy.optimize.nnls`); el residu és
     la distància mínima entre la "millor imitació possible" i la diana.
     Parar quan el residu supera el soroll del sensor.
   - **Eina 2 (`discrimination.py`)**: panel que millor separa dues classes
     (p. ex. sa vs malalt) a partir de perfils GC-MS reals, projectats amb
     el mateix model directe. Mètrica per defecte: ràtio de Fisher
     multivariant; opcionalment, accuracy amb validació creuada
     (`scikit-learn`).

El bucle greedy (`greedy_core.py`) és compartit i implementa les tres
parades del document: `target_crossed` (èxit), `diminishing_returns`
(sensors redundants) i `library_exhausted` (veredicte honest: la teva
llibreria de proteïnes no basta).

## Estructura

```
panel_selector/
    model.py           SensitivityMatrix + forward_response (nucli compartit)
    greedy_core.py      bucle greedy compartit
    specificity.py       Eina 1 — residu NNLS, cerca greedy, Monte Carlo opcional
    discrimination.py    Eina 2 — Fisher ratio / CV accuracy, cerca greedy
examples/
    example_sensitivity.csv       dades sintètiques (reprodueixen l'exemple
                                    numèric a mà del PDF: DcitOBP3+OBP-B,
                                    residu ≈0.171 amb MeSA vs
                                    nonanal/hexanol/limonè)
    demo_tool1_specificity.py     reprodueix pas a pas els números del PDF
    demo_tool2_discrimination.py  sa vs malalt sintètic, evita un OBP
                                    "decoy" fort però no discriminant
tests/
    test_core.py         verifica els números clau contra el document
```

## Ús ràpid

```bash
pip install numpy scipy pandas scikit-learn   # scikit-learn és opcional (només per cv_accuracy)
PYTHONPATH=. python examples/demo_tool1_specificity.py
PYTHONPATH=. python examples/demo_tool2_discrimination.py
PYTHONPATH=. python -m pytest tests/ -v
```

```python
from panel_selector import SensitivityMatrix, greedy_panel_for_specificity

S = SensitivityMatrix.from_ki_csv("Compound_OBP_binding.csv")  # format estil SCENTINEL
result = greedy_panel_for_specificity(
    S, target="MeSA", interferents=["nonanal", "1-hexanol", "linalool", "alpha-pinene"],
    noise=0.05,
)
print(result.panel, result.stop_reason)
```

## Connexió amb SCENTINEL v3

Aquest mòdul **no** integra amb el codi de SCENTINEL, però és compatible amb
les mateixes dades d'entrada: si tens `Compound_OBP_binding.csv` (Ki,
OBP × compost) pots carregar-lo directament amb `SensitivityMatrix.from_ki_csv`.
La secció "Limitations and future directions" del wiki de SCENTINEL ja
apunta cap aquí — "extending the framework from scoring single proteins to
scoring panels" — així que aquest mòdul és un primer prototip d'aquesta
segona direcció, fet des de zero perquè puguis provar-lo i iterar-hi abans
de decidir si (i com) es fon amb el pipeline principal.

## Limitacions (honestes, com diu el document)

- L'Eina 1 només et protegeix contra els interferents que **tu** hagis
  posat a la llibreria; un compost desconegut, fora de la llista, no hi
  està cobert.
- El model directe assumeix règim lineal (sense saturació tipus Langmuir).
- La qualitat de tot depèn de S: si les sensibilitats vénen de docking
  sense validar, cal calibrar-les amb algunes mesures reals.
