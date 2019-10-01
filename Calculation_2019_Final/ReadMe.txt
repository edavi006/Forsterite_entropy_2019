This repository contains all of the scripts and files needed to recreate all of the calculations done in the Silicate melting and vaporization paper (2019). The order of operations is given here.

1: Riemann_integral_Volumes_V2.py - Calculates volume of the release state using measurements of Us (shock velocity) in the sample and window. Monte Carlo uncertainty analysis is used here to propagate uncertainty to the partially released volumes.
Input - Z_Shot_Record-Deep Release-FO.csv
Output - Plot PDFs, and Release_VandP_Data.txt
Estimated Calc Time - 45 minutes.

2: Deep_release_independent_parallel.py - Calculates isentropes and associated gammas in an iterative fashion. Current set up is as in the paper, with a Gaussian formulation to calculate the release isentropes. After 300 iterations, the current monte carlo parameter set is deemed unconverging, and the perturbations are set to NaNs. Gaussian Formulations are not extremely flexible, so the NaN warning tends to show up a lot at density extremes for each pertubation step. This is part of the reason it takes so long to calculate, we still need enough usable isentropes to do statistics with.
Function Call - Parallel-MieGrun.py - actually performs the calculation
Input - Release_VandP_Data.txt
Output - Plot PDFs, P_V_E_releasePaths_gau.txt, and Gamma_Release_Data_gau.txt
Estimated Calc Time - 8 hours

3: Gamma_SwiftKrausMethod.py - Fits existing data with Gammas calculated in script #2. Uses Monte carlo perturbations to fit to the exponential/altschuler function as described in text. Calculated Co-Variance matrix to the parameters. 
Input - Gamma_Release_Data_gau.txt
Output - Plot PDFs and Co-variance matrix plus parameters given in the console.
Estimated Calc Time - 10 minutes

4: Hugoniot_entropy_V3_releaseGamma.py - Calculates entropy up to the isentrope, the isentrope temperature gradient, the intersections point between the principal Hugoniot and said isentrope, and then entropy on the Hugoniot. Plots generated from this script were not used in the paper. 
Input - Covariance Matrices, fits, and parameters for forsterite Hugoniot (Us-Up), (Us-T), and (gamma-rho).
Output - Z_Hugoniot.txt, Z_Hugoniot_us-up .txt, Plot PDFs, Isentrope intersection and additional step values are output to the console. 
Estimated Calc Time - 30 minutes

5: VaporFractions_ImpactVelocity.py and VaporFractions_ImpactVelocity_Hot_forsterite.py - Calculates pressures and impact velocities for specified vapor fractions and complete melting. Monte Carlo for uncertainty analysis.
Input - Z_Hugoniot.txt, Entropies of melting and vaporization (manually input), and entropy offset for Hot version, also needed is PulledHugoniots/L_VDome_STS_f.csv and forsterite-maneos_py.txt for comparison
Output - VaporFractions_Forsterite_298.txt or VaporFractions_Forsterite_1200.txt
Estimated Calc Time - 15 minutes

6: VaporFractions_ImpactVelocity_quartz.py and VaporFractions_ImpactVelocity_Hot_quartz - As above but for quartz.
Input - Silica-dome for vapor dome, and entropy offset for warm hot version.
Output - VaporFractions_Quartz_298.txt or VaporFractions_Quartz_1200.txt
Estimated Calc Time - 15 minutes

