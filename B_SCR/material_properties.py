
# ##WB IMPORTS
import os
import numpy as np
import math as m
from sklearn.linear_model import LinearRegression
from B_SCR.plots import *

def convert_fvd_engss(df=None, geometry=None, paths=None):
    """ FUNCTION TO CONVERT:
    FORCE (kN) TO ENGINEERING STRESS (MPa)
    DISPLACEMENT (mm) TO ENGINEERING STRAIN (mm/mm).
    IDENTIFY UTS: USE STRESS, GET FORCE, DISPLACEMENT, STRAIN
    Return:
        maximum displacement of experiment
        strain array up to uts
        stress array up to uts
        dictionary of uts data
    """
    force = df['FORCE']
    disp = df['DISPLACEMENT']
    # ##GET MAX DISPLACEMENT
    max_disp = round(df['DISPLACEMENT'].iloc[-1],2)

    # ##GET ALL ENGINEERING STRESS-STRAIN DATA (INCLUDING POST UTS)
    stress = np.divide(force, m.pi * (float(geometry['GAUGE_DIAMETER']) / 2) ** 2) * 1000
    strain = np.divide(disp, float(geometry['GAUGE_LENGTH']) / 2)
    # ##ADD STRESS STRAIN TO DF
    df['STRAIN']=strain
    df['STRESS']=stress
    # ##PLOT ENGINEERING DATA
    eng_stress_eng_strain(x=strain,
                          y=stress,
                          **paths)
    # ##IDENTIFY UTS
    uts_ind = stress.argmax()
    # ##GET THE UTS AS SERIES
    uts_df = df.iloc[uts_ind]
    new_ind=['UTS_%s'%(i) for i in uts_df.index]
    uts_df.index=new_ind
    uts_dic=uts_df.to_dict()
    # ##REDUCE THE ARRAYS SO THAT ONLY DATA PRIOR TO UTS IS RETURNED
    stress = stress[:uts_ind+1]
    strain = strain[:uts_ind+1]

    return max_disp, stress, strain, uts_dic


# Used to convert the LD Data to True Stress and Plastic Strain
def true_stress_strain(eng_stress=None, eng_strain=None):
    """ FUNCTION TO CALCULATE
    TRUE STRESS FROM ENGINEERING STRESS
    TRUE STRAIN FROM ENGINEERING STRAIN"""
    true_strain = np.log(1 + eng_strain)
    true_stress = eng_stress * (1 + eng_strain)
    return true_strain, true_stress


def aoc_calc_slope(true_strain, true_stress):
    """ function to calculate the slope of the extrapolated true stress-strain curve.
    function returns the minimum and maximum potential slopes based on five data points
    preceeding the UTS position. """
    # ##INTERESTED IN SLOPE BASED ON UTS AND PRECEEDING 5 POINTS
    # ##CAN'T EXTRAPOLATE BASED ON A SINGLE DATA POINT SO RANGE
    # ##MUST GOT FROM TWO TO SEVEN
    for i in range(2, 7, 1):
        stress = true_stress[-i:].values.reshape(-1, 1)
        strain = true_strain[-i:].values.reshape(-1, 1)
        model = LinearRegression().fit(strain, stress)
        c_slope = model.coef_[0][0]
        # ##FOR FIRST ITERATION SET BOTH MIN AND MAX VALUES TO CURRENT SLOPE
        if (i == 2):
            min_slope = c_slope
            max_slope = c_slope
        # ## FOR ALL OTHER ITERATIONS MODIFY THE MIN AND MAX
        elif (i > 2):
            if c_slope > max_slope:
                max_slope = c_slope
            elif c_slope < min_slope:
                min_slope = c_slope

    return min_slope, max_slope


def aoc_plastic_prop(strain, stress, yield_stress, modulus, **uts_dic):
    '''CALCULATE Y-INTERCEPT FROM STRESS,
        PLASTIC STRAIN AND SLOPE'''
    stress = stress.tolist()
    strain = strain.tolist()

    # ##FIND YIELD POINT AND IGNORE ALL VALUES PREVIOUS TO THAT POINT
    ind = (abs(stress-yield_stress)).argmin()
    stress = stress[ind:]
    strain = strain[ind:]

    # ##CALCULATE PLASTIC STRAIN FROM TRUE STRAIN
    plastic_strain = strain - (stress / modulus)
    # ##FIRST PLASTIC STRAIN VALUE IN ABAQUS MUST BE ZERO SO WE NEED TO ZERO THE ARRAY
    plastic_strain = plastic_strain - plastic_strain[0]
    # ##ONLY POSITIVE STRAINS ALLOWED - INDEX NEGATIVE VALUES AND REMOVE FROM BOTH STRESS AND STRAIN ARRAYS
    pos_vals = [(round(stress[i], 3), plastic_strain[i]) for i, val in enumerate(plastic_strain) if val >= 0]
    uts_dic['UTS_TRUE_STRESS']=pos_vals[-1][0]
    uts_dic['UTS_PLASTIC_STRAIN'] = pos_vals[-1][1]

    return pos_vals, uts_dic

def slope_range_calc(min_m, max_m, num_inc):
    inc_val = (max_m - min_m) / num_inc
    return np.arange(min_m, (max_m + inc_val), inc_val).tolist()