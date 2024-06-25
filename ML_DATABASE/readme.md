# Database of bayesian optimisation results for material P91_20_1 

## Experimental data: 
EXP_EXTRAP.csv: Displacement versus load for the physical experimental test (curve against which FEA data are compared)

## FEA data:
Main bayesian optimisation results are located in OUTPUT.csv

COLUMNS: 
JOB NUMBER: integer value describing job number
Q1: GTN parameter Q1 value used in analysis
Q2: GTN parameter Q2 value used in analysis
Q3: GTN parameter Q3 value used in analysis
EN: GTN parameter EN value used in analysis
SN: GTN parameter SN value used in analysis
FN: GTN parameter FN value used in analysis
F: GTN parameter F value used in analysis
M: GTN parameter M value used in analysis
ORIG_MAPE: Mean average percentage error used to describe how the FEA curve compares to the experimental data 
MAPE1: Mean average percentage error calculated for the FEA curve up to UTS value
MAPE2: Mean average percentage error calculated for the FEA curve after UTS value
WMAPE: Weighted mean average percentage error
KAPPA: Kappa value used to control exploration/exploitation in bayesian algorithm. 
SIM_TIME: Approximate time for FEA simulation to complete. Note: incorrect values for jobs 1-12. 

### Individual FEA result files:

Note: individual jobs are tracked using the integer 'job number' value. The location of this value is demonstrated below using an asterisk *

ABA_JOB*.csv: True stress-plastic strain material properties used for FEA
JOB*_ENERGY.csv: Internal and external energy values from FEA. Not relevant to this particular analysis
JOB*_JSON.txt: Dictionary of material properties and file paths used to track individual values for jobs
JOB*_LD_DATA.csv: Displacement versus load from FEA analysis (raw form)
JOB*_SIM_EXTRAP.csv: Modified version of JOB*_LD_DATA.csv where data are interpolated and potentially smoothed to match acquisition rate of experimental data. This file is used to compare with experimental data.  
JOB*_SIM_EXTRAP_NOSMOOTH.csv: Modified version of JOB*_LD_DATA.csv where data are interpolated to match acquisition rate of experimental data, no smoothing function applied.
JOB*_SIM_EXTRAP_SMOOTH.csv: Modified version of JOB*_LD_DATA.csv where data are interpolated to match acquisition rate of experimental data, data is smoothed to remove noise (should be identical to JOB*_SIM_EXTRAP.csv).
JOB*_NECKING.csv: Specimen diameter as function of displacement and time. Used to compare whether plasticity is accurate or not (not relevant to current analysis). 



