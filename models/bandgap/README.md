# Public optical-gap baseline

Source: Jacobsson et al., The Perovskite Database archive v1.0.1, DOI https://doi.org/10.5281/zenodo.5837035 ; release paper https://doi.org/10.1038/s41560-021-00941-3 . Data licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .

Changes: filter to explicit 3D Pb-based Cs/MA/FA-I/Br compositions with Absorption/EQE/Tauc measurement basis; exclude composition-derived labels; normalize valid stoichiometry; aggregate repeated DOI/composition records by median; fit a Ridge polynomial selected on publication-grouped validation only. Original raw CSV SHA-256 and frozen rules are recorded in the report.

697 eligible source rows become 149 DOI/composition records from 133 papers and 36 compositions. Primary holdout has 30 records / 27 papers. Separate composition holdout has 14 records. Family test counts for CsMAFAPbIBr, CsFA and MAFA are too small for activation, so those models remain unavailable. Current deployment keeps the test set out of fitting.

Reproduce with `python scripts/train_bandgap_baseline.py --source <original-csv> --output <new-directory>` . The full CSV is available from the cited archive. Do not substitute estimated bandgaps or synthetic predictions for measured targets.
