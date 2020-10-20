# LWA-1D
Code of the 1D model of local wave activity in the midlatitudes based on work by Nakamura and Huang [here](https://science.sciencemag.org/content/early/2018/06/21/science.aat0721). This model was used in an upcoming paper by Claire Valva (git author) and Noboru Nakamura at the University of Chicago. 

This version of the model was originally written by Paradise et al. 2019 for use in this [paper](https://doi.org/10.1175/JAS-D-19-0095.1) — portions, primarily the spectra of transient forcing, has since been edited. 

The executable file (run_traffic_model_cs.py) takes 3 arguments:
1. a2Lambda: $2*\alpha*Y1$ (integer), typically 11, this is how the paramter $\alpha$ is adjusted.
2. gamma: typically 3 or 4 (float), strength of the transient eddies
3. Uj: background speed of the jet $ms^{-1}$ (integer), typtically $60ms^{-1}$. 

The file, arr =  sum_sds_avgs_01.npy, is a numpy array which contains mean 2d-fft statistics for $45^\circ$ N. The relevant quantities are: the mean magnitude of a wavenumber/frequency pair is arr\[1,2, wavenumber, frequency\] and the standard deviation of that magnitude which is arr\[0,2, wavenumber, frequency\]. 

This model is set to run for 3 years, repeating 200 times. The saving filenames will need to be adjusted (found at the end of run_traffic_model_cs.py) depending on the computer used. 

For questions about the model, one can either submit a pull request or send me (Claire Valva) an email at claire + v + (at) + nyu + .edu. 


