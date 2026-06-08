# IDR CSV export

These CSV files reconstruct the existing evaluator's `calc_IDR()` result for the
CRF sequence experiments.

- `idr_reconstructed_details_all.csv`: one row per vertebra prediction.
- `idr_reconstructed_per_image_all.csv`: one row per image and variant.
- `idr_reconstructed_summary.csv`: metric summary for each experiment and variant.
- `*__idr_details.csv`: convenient per-experiment subsets.

Important: `accepted_by_old_calc_IDR` reproduces the current evaluator behavior.
It checks whether the signed mean bbox coordinate difference is below 2. It does
not use absolute distance or IoU, so negative differences can be accepted even
when the category is wrong. Use `old_IDR_false_accept` to audit these cases.

`position_match_IDR2` is the stricter sequence-position category match used by
the current `IDR2` calculation.
