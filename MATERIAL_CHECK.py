import os
import pandas as pd
import numpy as np
import copy
import collections
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 6)
##set font size
font={'family': 'sans-serif',
      'weight': 'normal',
      'size':14}
plt.rc('font', **font)
from B_SCR.general_functions import load_json_file, write_json_file, merge_dicts, new_dir_add_dic
from B_SCR.material_properties import convert_fvd_engss, true_stress_strain, aoc_calc_slope
from B_SCR.plots import eng_stress_eng_strain, plot_sec_der_peaks, compare_interp_true

""" 
CHECK MATERIAL ASSESSMENT IS BEING PREFORMED IN A REASONABLE WAY
1. CONVERT P-DELTA TO ENG STRESS - ENG STRAIN
2. CONVERT ENG STRESS - ENG STRAIN TO TRUE STRESS - TRUE STRAIN
2. CALCULATE SLOPES FROM TRUE STRESS - TRUE STRAIN
3. EXTEND ENG STRESS - ENG STRAIN USING SLOPES 
4. RUN ABAQUS FOR EACH SIMULATION - NO DUCTILE DAMAGE.
5. COMPARE ORIGINAL P-DELTA TO SIMULATION RESULTS.
"""

# ##CREATE DICTIONARY OF PATHS TO RELEVANT DIRECTORIES AND FILES
path_dic = {'cwd': os.getcwd(),
            'project_dir': os.path.join(os.getcwd()),
            'raw_data': os.path.join(os.getcwd(), 'A_RAW_DATA'),
            'build': os.path.join(os.getcwd(), 'B_SCR/aba_build_nodamage.py'),
            'postp': os.path.join(os.getcwd(), 'B_SCR/aba_pp.py')}

# ##DICTIONARY HOLDING GEOMETRICAL DATA
geom_dic = {'GAUGE_LENGTH': 25.,
            'GAUGE_DIAMETER': 4.,
            'CONN_DIAMETER': 5.,
            'SPECIMEN_LENGTH': 72.,
            'ROUND_RADIUS': 3.,
            'THREADED_LENGTH': 15}
# ##HEADERS FOR CSV OUTPUT FILE
csv_output_headers = ['JOB_NUM', 'M', 'SIM_TIME', 'MAPE1', 'MAPE2']

# ##LIST ALL FILES IN OUR RAW DATA FOLDER (PROVIDE MATERIAL NAMES)
material_list = [f for f in os.listdir(path_dic['raw_data']) if not 'DOE_ARRAY.csv' in f]
# material_list = ['P91_20_1.csv', 'P91_500.csv']
for i, material in enumerate(material_list):
    # ##ADD EXPERIMENTAL DATA TO PATH DIC
    path_dic['exp_fvd'] = os.path.join(path_dic['raw_data'], material)
    material = material[:-4]
    print('MATERIAL: %s' %(material))
    # ##CREATE MATERIALS SUBDIRECTORY
    path_dic = new_dir_add_dic(dic=path_dic,
                               key='curr_results',
                               path=os.path.join(path_dic['cwd'], 'OUTPUT'),
                               dir_name=material,
                               exist_ok=True)
    #################################
    ## MATERIAL ASSESSMENT
    ################################
    # ##READ IN THE RAW P-DELTA DATA
    fvd = pd.read_csv(path_dic['exp_fvd'], header=[0, 1]).droplevel(level=1, axis=1)
    # ##CALCULATE ENGINEERING STRESS-STRAIN
    eng_stress, eng_strain, uts_dic = convert_fvd_engss(df=fvd,
                                                        geometry=geom_dic,
                                                        paths=path_dic)
    # ##ADD RAW DATA PATH TO UTS DIC
    uts_dic['RAW_DATA'] = path_dic['exp_fvd']
    # ##CALCULATE TRUE STRESS AND TRUE STRAIN FROM ENG STRESS-STRAIN
    true_strain, true_stress = true_stress_strain(eng_stress=eng_stress,
                                                  eng_strain=eng_strain)
    # ##WE KNOW LAST VALUES OF TRUE STRESS/STRAIN REPRESENT UTS
    uts_dic = merge_dicts(uts_dic, {'TRUE_STRAIN':true_strain.iloc[-1],
                                    'TRUE_STRESS':true_stress.iloc[-1]})
    # ##CALCULATE THE SLOPE OF THE CURVES FROM TRUE STRESS-STRAIN
    min_slope, max_slope = aoc_calc_slope(true_strain,
                                          true_stress)
    # ##CALCULATE SECOND DERIVATIVE
    # ##ASSUME YIELD STRESS CAN ONLY OCCUR PRIOR TO 0.2% STRAIN = STRAIN OF .002
    strain_idx = np.where(true_strain <= 0.002)
    # ##DATA ARE VERY SPARSE SO WE'RE GOING TO INTERPOLATE BETWEEN EACH POINT OF THE CURVE
    # ##USING A SPLINE
    func = interp1d(true_strain.values, true_stress.values, kind='linear')
    interp_strain = np.arange(true_strain.iloc[0], true_strain.iloc[-1], 1e-4)
    interp_stress = func(interp_strain)
    # ##COMPARE ENGINEERING WITH INTERPOLATED
    compare_interp_true(truex=true_strain,
                        truey=true_stress,
                        interpx=interp_strain,
                        interpy=interp_stress,
                        kind=func._kind,
                        **path_dic)
    # ##ITERATE RANGE OF WINDOW SIZES AND RETURN THE WINDOW SIZE THAT GIVES THE BEST R2 SCORE
    sav_dic={}
    for j, wl in enumerate([x for x in np.arange(11, 101, 3) if x % 2 != 0]):
        # ##LIMIT SEC DER CALCULATION TO EARLY REGION ONLY
        sder_strain = savgol_filter(interp_strain, window_length=wl, polyorder=3, deriv=2)
        # ##INDEX OF POINT CLOSEST TO ZERO
        zero = np.abs(sder_strain - 0.0).argmin()
        # ##CALCULATE MODULUS
        mod_strain = interp_strain[:zero].reshape(-1, 1)
        mod_stress = interp_stress[:zero].reshape(-1, 1)
        if len(mod_strain) > 5:
            model = LinearRegression().fit(mod_strain, mod_stress)
            # ##GET Y PREDICTION
            prediction = model.predict(mod_strain)
            sav_dic[wl] = {'E': model.coef_[0][0],
                           'SIGMA_Y':interp_stress[zero],
                           'r2': round(r2_score(mod_stress, prediction), 3),
                           'SEC_DER':sder_strain,
                           'ZERO':zero,
                           'WINDOW_LENGTH':wl}
    for k in sav_dic.keys():
        # ##PLOT FOR EACH WL ANALYSIS
        plot_sec_der_peaks(true_strain=true_strain,
                           true_stress=true_stress,
                           interp_strain=interp_strain,
                           interp_stress=interp_stress,
                           img_name='SEC_DER_%s'%(k),
                           data_dic=sav_dic[k],
                           **path_dic)
    # ##CREATE DF FROM DICTIONARY AND TRANSPOSE
    df_dic = pd.DataFrame(copy.deepcopy(sav_dic)).transpose()
    # ##GET R2 GREATER THAN 0.9 ONLY
    df_dic = df_dic[df_dic['r2']>=0.95]
    df_dic['MULTI'] = df_dic['r2'] * df_dic['SIGMA_Y']
    # df_dic = df_dic[df_dic['r2']>= df_dic['r2'].max()*0.9].sort_values(['r2'], ascending=False).iloc[0:3].dropna(axis=1)
    # ##SORT DF_DIC BY YIELD STRENGTH
    df_dic.sort_values(['MULTI'], ascending=[False], inplace=True)
    # ##SELECT MAXIMUM YIELD VALUE AS 'BEST' VALUE
    best_key = df_dic.iloc[0].name
    best = sav_dic[best_key]
    # ##ADD YOUNG'S MOD AND YIELD STRENGTH TO THE UTS DICTIONARY
    uts_dic = merge_dicts(uts_dic, {'E':best['E'], 'SIGMA_Y':best['SIGMA_Y']})
    # ##PLOT THE SECOND DERIVATIVE OUTPUTS (INC MODULUS LINE)
    plot_sec_der_peaks(true_strain=true_strain,
                       true_stress=true_stress,
                       interp_strain=interp_strain,
                       interp_stress=interp_stress,
                       img_name='SEC_DER',
                       data_dic=best,
                       **path_dic)
    # #################################
    # ## EXTEND DATA BEYOND UTS
    ###################################
    # ##RANGE OF SLOPES
    m_range = np.arange(0, max_slope + 50, 50)
    # ##EXTEND TRUE STRESS - TRUE STRAIN USING SLOPES
    slope_dic={}
    for j, m in enumerate(m_range):
        # ##THE INTERCEPT IS A FUNCTION OF SIGMA TRUE UTS AND
        # ##EPSILON TRUE UTS (C = SIG_T - M*ESP_T)
        c = uts_dic['TRUE_STRESS'] - (m * uts_dic['TRUE_STRAIN'])
        slope_dic[m]={'Y_INTERCEPT':c,
                      'ABAQUS_PLASTIC':os.path.join(path_dic['curr_results'], 'ABA_M%s.csv' % (str(int(m))))}
        # ##SET STRAIN RANGE TO BE AT LEAST 1000 ELEMENTS IN SIZE
        estrain = np.linspace(interp_strain[-1] + 1e-4, 2, num=1000).reshape(-1, 1)
        estress = m * estrain + c
        # ##COMBINE ORIGINAL AND EXTENDED DATA TO GET THE FULL MATERIAL PROPERTIES FOR ABAQUS
        df = pd.DataFrame(data={'TRUE_STRAIN':np.concatenate((interp_strain, estrain.flatten()), axis=0),
                                'TRUE_STRESS':np.concatenate((interp_stress, estress.flatten()), axis=0)})
        # ##ABAQUS REQUIRES STRAIN AND STRESS TO START FROM YIELD POSITION
        # ## WE NEED TO MODIFY THE TRUE STRESS - TRUE STRAIN TO TRUE STRESS - PLASTIC STRAIN
        plastic = df[df['TRUE_STRESS']>=best['SIGMA_Y']].copy()
        # ##MODIFY STRAIN TO BE ZERO AT YIELD STRESS
        plastic['PLASTIC_STRAIN'] = plastic['TRUE_STRAIN'] - (plastic['TRUE_STRESS'] / best['E'])
        # ##RESET STRAIN TO BE ZERO AT YIELD
        plastic['PLASTIC_STRAIN'] = plastic['PLASTIC_STRAIN'] - plastic['PLASTIC_STRAIN'].iloc[0]
        # ##REPLACE ANY NEGATIVE STRAINS WITH VERY LOW STRAIN
        plastic['PLASTIC_STRAIN'] = np.where(plastic['PLASTIC_STRAIN']<0, 1E-20, plastic['PLASTIC_STRAIN'])
        # ##DROP TRUE STRAIN
        plastic.drop('TRUE_STRAIN', axis=1, inplace=True)
        # ##SAVE ABAQUS DATA TO CSV FILE
        plastic.to_csv(os.path.join(path_dic['curr_results'], 'ABA_M%s.csv' % (str(int(m)))), index=False)
    # ##BUNDLE SLOPE DICTIONARY INTO UTS_DICTIONARY - TRACK 'M', Y_INTERCEPT AND ABAQUS PLASTIC VALUES
    uts_dic = merge_dicts(uts_dic, {'SLOPE':slope_dic})
    # ##WRITE UTS DIC TO JSON
    write_json_file(dic=uts_dic, pth=path_dic['curr_results'], filename=material + '_properties.txt')
    plt.close('all')