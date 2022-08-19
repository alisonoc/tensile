The purpose of this module is to analyse material properties that are planned for use in bayesian optimisation sections.

Here the following procedure is applied:
1. Read in load-displacement test data from raw data folder.
    A. Two columns of data: Force, Displacement.
    A. Units: force kN, displacement in mm [Giving stress in MPa].
2. Calculate engineering stress-strain data.
    A. Identify UTS data and maximum displacement
3. Calculate true stress-strain data.
4. Calculate the maximum potential slope of the line (use up to 5 points prior to UTS value).
5. Identify linear regression data (yield strength and Young's modulus):
    A. Using second derivative of strain and various filtering properties:
        I. Cycle the filter window length
    B. Log all outputs in dictionary
    C. Use the dictionary to identify the most appropriate yield strength value:
        I. R2 must exceed 0.95
        II. Calculate the relative error r2/(sigma_y*mape) for each window
        III. Sort values by the relative error and select first row (best combined r2, yield, error)
6. Plot the 'best' dictionary output
7. Merge UTS_DIC with best dictionary
8. For all potential slopes between 0 and maximimum (see point 4) in increments of 50 (we assume changes in slope of <50 MPa has negligible effect):
    A. Calculate the y-axis intercept (c from equation of line: y=mx+c)
    B. Create array of strains using linspace (1000 elements, stop=final interpolated strain value)
    C. Create array of stresses using equation of line (y=mx+c)
    D. Create df of true strain(7B), true stress (7C) [concate to interpolated strain and stress]
    E. Limit df (7D) to stresses greater than or equal to yield strength (5CIII)
    F. Calculate plastic strain (epsilon_p = epsilon_t - sigma_t/E)
    G. Add row to df representing first line (yield stress=5CIII, epsilon_p=0.00) for abaqus
    H. Export df to csv file *This csv file has plastic strain versus true stress for Abaqus input)
    I. Dictionary slope value:{y_intercept, filepath to abaqus plastic data}
9. Merge uts_dic with 8I dictionary. Export to json.