# VineSplit input table

One row per compound, these columns only:

| CAS | name | mean healthy | SE healthy | mean diseased | SE diseased |
|---|---|---|---|---|---|
| 119-36-8 | methyl-salicylate | 177.98702576174142 | 82.53198760851039 | 1449.3125096904478 | 231.04735355632226 |
| 928-95-0 | (E)-2-Hexenol | n.d. | n.d. | 41.59926480663321 | 5.765700229487674 |

- **CAS** is mandatory (it is how the compound is found in `Compound_OBP_binding.csv`).
- **name** is optional.
- **SE** = standard error of the mean, as published. The number of plants per group is given with `--n` (default 3).
- **n.d. is fine: the program reads it as 0.**

Template: `templates/vinesplit_input.xlsx` (or any `.csv` with the same header).

```bash
python scentinel_tool2_discrimination.py --profiles templates/vinesplit_input.xlsx --source experimental
```
