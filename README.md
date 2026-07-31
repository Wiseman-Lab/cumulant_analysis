# Analysis code for "Quantifying GPCR - G protein binding affinity in living cells via fluorescence image cumulant analysis"

Starting from two-channel tiff time series, a typical workflow will use the following scripts and notebooks

- imageprep.ipynb – image segmentation
- batch_get_cumulants.py - segmented image cumulant calculations
- brightness_fit.ipynb - fluorophore brightness estimation
- Batch_run_Lsfit.py - 3 species two colour cumulant analysis for interaction number density calculations
- FP_fit_script.py - Fokker-Planck model fitting
- Check_results.ipynb - results verification
- Combine_results.ipynb - convenience plotting

