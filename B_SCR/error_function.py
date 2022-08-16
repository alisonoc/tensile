import os
import pandas as pd
import numpy as np
from scipy import interpolate


def mean_abs_pe(target, forecast):
    ''' FUNCTION TO CALCULATE MEAN ABSOLUTE PERCENTAGE ERROR (MAPE).
	FIRST CALCULATE THE MAPE FOR EACH POINT IN THE SERIES
	OVERALL MAPE IS ASSUMED TO BE THE AVERAGE.'''
    # ##CALCULATE THE MAPE FOR EACH POINT IN THE SERIES
    series_mape = 100 * abs(((target - forecast) / target))
    # ##GET THE AVERAGE MAPE OVER THE SERIES SIZE
    total_mape = np.sum(series_mape) / series_mape.size
    return total_mape

def split_df_by_uts(df=None, uts=None):
    """ FUNCTION TO RETURN DF VALUES UP TO UTS ONLY"""
    # ##FIND CLOSEST MATCH TO UTS FORCE
    ind = df['RF'].sub(uts).abs().idxmin()
    pre = df[df.index <= ind]
    post = df[df.index >= ind]
    return pre, post

def interpolate_data(df, min_inc):
    ''' FUNCTION TO INTERPOLATE DATA TO INCLUDE MINIMUM
	INCREMENT VALUE. RETURNS DF BASED ON INTERPOLATED VALUES.'''
    # ##REMOVE NON UNIQUE X VALUES
    df = df.drop_duplicates(subset=['U'])
    # ##IDENTIFY X, Y SERIES
    y = df['RF']
    x = df['U']
    # ##INTERPOLATE THE DATA
    func = interpolate.interp1d(x, y, kind='cubic')
    # ##EXTEND X SERIES TO HOST MINIMUM INCREMENT
    new_x = np.arange(x.values[0], x.values[-1], min_inc)
    # ##ESTIMATE Y VALUES FOR NEW X
    new_y = func(new_x)
    # ##CREATE NEW DF BASED ON NEW X/Y
    new_df = pd.DataFrame(np.column_stack((new_x, new_y)), columns=['U', 'RF']).drop_duplicates()
    # ##RETURN THE NEW DF
    return new_df


def read_sim_results(result_path=None, exp_path=None, pad_value='zeros'):
    ''' READ SIMULATION FVD AND EXP FVD
	FIND MINIMUM INCREMENT IN EITHER SIMULATION OR EXP DATA
	FIT FORCE VALUES TO MINIMUM DISPLACEMENT INCREMENT.
	aba_dir: directory where abaqus files are located
	result_path: the filepath of the simulation results,
	exp_fname: filepath for experimental results
	neck_diameter: the diameter of necked region in simulated results'''
    # ##READ SIMULATION DATA
    sim_df = pd.read_csv(os.path.join(result_path))
    # ##CONVERT SIM FORCE FROM NEWTONS TO KN
    sim_df['RF'] = sim_df['RF'] / 1000
    # ##READ EXPERIMENT DATA
    exp_df = pd.read_csv(exp_path, names=['U', 'RF'], skiprows=2)
    # ##FIND THE MINIMUM INCREMENT IN EITHER SIM_DF OR EXP_DF
    sim_inc = sim_df['U'].diff().replace(0, np.nan).min()
    exp_inc = exp_df['U'].diff().replace(0, np.nan).min()
    min_inc = round(min(sim_inc, exp_inc), 6)
    # ##INTERPOLATE THE FUNCTIONS USING SMALLEST INCREMENT
    sim_df = interpolate_data(sim_df, min_inc)
    exp_df = interpolate_data(exp_df, min_inc)
    # ##IF DF HAVE DIFF DISPLACEMENTS APPEND PAD SIMULATED DATA
    if sim_df['U'].iloc[-1] <= exp_df['U'].iloc[-1]:
        sim_df = pad_sim_data(exp_df=exp_df,
                              sim_df=sim_df,
                              pad_value=pad_value)
    return sim_df, exp_df


def pad_sim_data(exp_df=None, sim_df=None, pad_value='zeros'):
    """ FUNCTION TO PAD SIMULATED DATA SO THAT DISPLACEMENT OF SIMULATION
    REACHES SAME DISPLACEMENT SEEN IN EXPERIMENT.
    EXP_DF: EXPERIMENTAL DATAFRAME
    SIM_DF: SIMULATION DATAFRAME
    PAD_VALUE: FORCE VALUE TO BE PAD
    """
    if pad_value=='last':
        pvalue=sim_df['RF'].iloc[-1]
    elif pad_value=='zeros':
        pvalue=0.0
    # ##WE REALLY ONLY WANT TO PAD THE FORCE FUNCTION BUT THE DISPLACEMENT MUST REMAIN CONSISTANT WITH FORCE
    # ##PAD ROWS TO EXTEND TO SAME ARRAY SIZE AS EXPERIMENTAL DF
    pad_area = [(0, exp_df.shape[0] - sim_df.shape[0])]
    # ##DISP ARRAY LINEAR RAMP UP TO EXPERIMENTAL MAX
    new_disp = np.pad(sim_df['U'].to_numpy(),
                      pad_width=pad_area,
                      mode='linear_ramp',
                      end_values=exp_df['U'].iloc[-1])
    # ##PAD FORCE ARRAY
    new_force = np.pad(sim_df['RF'].to_numpy(),
                       pad_width=pad_area,
                       mode='constant',
                       constant_values=pvalue)
    # ##NEW DF FOR SIM
    sim_df = pd.DataFrame.from_dict({'RF': new_force,
                                     'U': new_disp},
                                    orient='columns')
    return sim_df
