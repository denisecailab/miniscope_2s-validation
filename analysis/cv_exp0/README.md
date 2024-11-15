# Analysis folder of dual channel imaging experiment

This is the master analysis folder for dual channel imaging experiment.
To reproduce all analysis, make sure you have `data` folder available.
Then, run all the `.py` scripts in order (scripts starting with the same number can be run in parallel).
To reproduce individual figures in the manuscript from intermediate results, check details in the section below.

## Reproducing individual figures

- Figure 3
    - Requires:
        - `intermediate/frame_label/behav.feat`
        - `intermediate/frame_label/behav_v4.feat`
    - Run:
        - `02.compare_behavior.py`
    - Output:
        - Panel B: `figs/behav_comparison/example.svg`
        - Panel C: `figs/behav_comparison/comparison.svg`
- Figure 4
    - Requires:
        - `intermediate/processed/green`
        - `intermediate/processed/red`
        - `log/sessions.csv`
        - `intermediate/cross_reg/red/mappings_meta_fill.pkl`
        - `intermediate/cross_reg/green/mappings_meta_fill.pkl`
    - Run:
        - `04.register_g2r.py`
    - Output:
        - Panel A: `figs/register_g2r/cells/m22-A_example.svg`
        - Panel B: `figs/register_g2r/overlap_ncell.svg`
        - Panel C: `figs/register_g2r/overlap_prop.svg`
- Figure 5
    - Requires:
        - `intermediate/processed/green`
        - `intermediate/processed/red`
        - `log/sessions.csv`
        - `intermediate/cross_reg/red/mappings_meta_fill.pkl`
        - `intermediate/cross_reg/green/mappings_meta_fill.pkl`
    - Run:
        - `04.register_g2r.py`
    - Output:
        - `figs/register_g2r/traces.svg`
- Figure 6
    - Requires:
        - `data/wavelength/fpbase_spectra_EGFP.csv`
        - `data/wavelength/et600-50m.txt`
        - `data/wavelength/et525-50m.txt`
    - Run:
        - `04.register_g2r.py`
    - Output:
        - Panel A: `figs/register_g2r/crosstalk_wavelength.svg`
        - Panel B: `figs/register_g2r/crosstalk_distribution.svg`
- Figure 7
    - Requires:
        - `intermediate/processed/green`
        - `intermediate/processed/red`
        - `log/sessions.csv`
        - `intermediate/cross_reg/red/mappings_meta_fill.pkl`
        - `intermediate/cross_reg/green/mappings_meta_fill.pkl`
    - Run:
        - `04.register_g2r.py`
    - Output:
        - `figs/register_g2r/summary_agg.svg`
- Figure 8
    - Requires:
        - `log/sessions.csv`
        - `intermediate/processed/green`
        - `intermediate/frame_label/fm_label.nc`
        - `intermediate/cross_reg/green/mappings_meta_fill.pkl`
        - `intermediate/cross_reg/red/mappings_meta_fill.pkl`
        - `intermediate/register_g2r/green_mapping_reg.pkl`
    - Run:
        - `05.drift_analysis.py`
    - Output:
        - Panel A: `figs/drift/actMean-place_cells.svg`
        - Panel B: `figs/drift/py_corr-place_cells.svg`


