import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.style as style
style.use('tableau-colorblind10')
plt.rcParams["figure.figsize"] = (6, 6)
##set font size
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)

import pandas as pd
import itertools

marker = itertools.cycle(('s', '+', 'o', '*', 'x', 'v', 'D'))


def plot_mape_v_sim_time(bayes_dic, **path_dic):
    fig, ax2d = plt.subplots()
    ax = np.ravel(ax2d)

    df = pd.DataFrame.from_dict(bayes_dic)

    for i, rw in df.iterrows():
        ax[0].scatter(rw['SIM_TIME'] / 60,
                      rw['MAPE'],
                      marker=next(marker),
                      label='JOB%s' % (int(rw['JOB_NUM'])))

    # AXES LIMITS
    ax[0].set_xlim([0, (max(df['SIM_TIME'] / 60)) * 1.25])
    ax[0].set_ylim([0, df['MAPE'].max() * 1.25])

    # AT LEAST FIVE TICK MARKS ON X AND Y AXES
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))

    # AXES LABELS
    ax[0].set_xlabel('Time, minutes')
    ax[0].set_ylabel('Mean absolute percentage error, %')
    # ##LEGEND
    ax[0].legend(bbox_to_anchor=(1.1, 1),
                 loc='upper left',
                 ncol=6,
                 borderaxespad=0)
    # save figure
    plt.savefig(os.path.join(path_dic['curr_results'],
                             'MAPE_V_SIM_TIME.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close('all')


def plot_mape_v_sim_num(bayes_dic, **path_dic):
    fig, ax2d = plt.subplots()
    ax = np.ravel(ax2d)

    df = pd.DataFrame.from_dict(bayes_dic)

    for i, rw in df.iterrows():
        ax[0].scatter(rw['JOB_NUM'],
                      rw['MAPE'],
                      marker=next(marker),
                      label='JOB%s' % (int(rw['JOB_NUM'])))

    # AXES LIMITS
    ax[0].set_xlim([0, (max(df['JOB_NUM'])) * 1.25])
    ax[0].set_ylim([0, df['MAPE'].max() * 1.25])

    # AT LEAST FIVE TICK MARKS ON X AND Y AXES
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))

    # AXES LABELS
    ax[0].set_xlabel('Iteration number')
    ax[0].set_ylabel('Mean absolute percentage error, %')
    # ##LEGEND
    ax[0].legend(bbox_to_anchor=(1.1, 1),
                 loc='upper left',
                 ncol=6,
                 borderaxespad=0)
    # save figure
    plt.savefig(os.path.join(path_dic['curr_results'],
                             'MAPE_V_SIM_NUM.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close('all')

def best_and_worst(bayes_dic, **path_dic):
    fig, ax2d = plt.subplots()
    ax = np.ravel(ax2d)

    # ##MAPE RESULTS
    df = pd.DataFrame.from_dict(bayes_dic)
    # ##USING PATH DIC FIND GET RELEVANT DATA
    exp_df = pd.read_csv(path_dic['exp_fvd'], header=[0, 1]).droplevel(1, axis=1)
    # ##CONVERT TO NEWTONS
    exp_df['RF'] = exp_df['FORCE'] * 1000
    # ##SORT USING MAPE COLUMN AND RESET INDEX
    sorted = df.sort_values(by=['MAPE'],
                            axis=0,
                            ascending=True).reset_index(drop=True)
    # ##BEST IS FIRST ROW, WORST IS LAST ROW (ASCENDING=TRUE
    best = sorted.iloc[0]
    worst = sorted.iloc[-1]
    try:
        best_df = pd.read_csv(path_dic['JOB%s_fvd' % (int(best.loc['JOB_NUM']))])
    except:
        best_df = pd.read_csv(os.path.join(path_dic['curr_results'],
                                           'JOB%s_LD_DATA.csv' % (int(best.loc['JOB_NUM']))))
    try:
        worst_df = pd.read_csv(path_dic['JOB%s_fvd' % (int(worst.loc['JOB_NUM']))])
    except:
        worst_df = pd.read_csv(os.path.join(path_dic['curr_results'],
                                           'JOB%s_LD_DATA.csv' % (int(worst.loc['JOB_NUM']))))

    # ##PLOT DATA
    ax[0].scatter(exp_df['DISPLACEMENT'],
                  exp_df['RF']/1000,
                  marker='.',
                  color='k',
                  label='Experimental')
    ax[0].plot(best_df['U'],
               best_df['RF']/1000,
               linestyle='--',
               label='Best simulation')
    ax[0].plot(worst_df['U'],
               worst_df['RF']/1000,
               linestyle=':',
               label='Worst simulation')
    # ##AXES TEXT
    ax[0].text(x=0.01,
               y=0.05,
               s='Best simulation. Job number: %s Error: %s %%\nWorst simulation. Job number: %s Error: %s %%'
                 %(int(best['JOB_NUM']), round(best['MAPE'], 2), int(worst['JOB_NUM']), round(worst['MAPE'], 2)),
               horizontalalignment='left',
               verticalalignment='center',
               transform=ax[0].transAxes)

    # AXES LIMITS
    ax[0].set_xlim([0, exp_df['DISPLACEMENT'].max() * 1.25])
    ax[0].set_ylim([0, exp_df['FORCE'].max() * 1.25])

    # AT LEAST FIVE TICK MARKS ON X AND Y AXES
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))

    # AXES LABELS
    ax[0].set_xlabel('Displacement, mm')
    ax[0].set_ylabel('Force, kN')
    # ##LEGEND
    ax[0].legend(bbox_to_anchor=(1.1, 1),
                 loc='upper left',
                 ncol=1,
                 borderaxespad=0)
    # save figure
    plt.savefig(os.path.join(path_dic['curr_results'],
                             'COMPARE_FVD.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close('all')


def exp_v_simulation(i_fea=None, mape=None, **path_dic):
    fig, ax2d = plt.subplots()
    ax = np.ravel(ax2d)

    # ##USING PATH DIC FIND GET RELEVANT DATA
    exp_df = pd.read_csv(path_dic['exp_fvd'], header=[0, 1]).droplevel(1, axis=1)
    # ##CONVERT TO NEWTONS
    exp_df['RF'] = exp_df['FORCE'] * 1000
    # ##USE PATH DIC TO FIND SIMULALTION
    sim_df = pd.read_csv(path_dic['JOB' + str(int(i_fea)) + '_fvd'])

    # ##PLOT DATA
    ax[0].scatter(exp_df['DISPLACEMENT'],
               exp_df['RF']/1000,
               color='k',
               label='Experimental')
    ax[0].plot(sim_df['U'],
               sim_df['RF']/1000,
               color='k',
               linestyle='--',
               label='simulation')
    # ##AXES TEXT
    ax[0].text(x=0.01,
               y=0.05,
               s='Job number: %s Error: %s %%'
                 %(int(i_fea), round(mape, 2)),
               horizontalalignment='left',
               verticalalignment='center',
               transform=ax[0].transAxes)

    # AXES LIMITS
    ax[0].set_xlim([0, exp_df['DISPLACEMENT'].max() * 1.25])
    ax[0].set_ylim([0, exp_df['FORCE'].max() * 1.25])

    # AT LEAST FIVE TICK MARKS ON X AND Y AXES
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))

    # AXES LABELS
    ax[0].set_xlabel('Displacement, mm')
    ax[0].set_ylabel('Force, kN')
    # ##LEGEND
    ax[0].legend(bbox_to_anchor=(1.1, 1),
                 loc='upper left',
                 ncol=1,
                 borderaxespad=0)
    # save figure
    plt.savefig(os.path.join(path_dic['curr_results'],
                             'COMPARE_FVD_JOB%s.png'%(str(int(i_fea)))),
                dpi=300,
                bbox_inches='tight')
    plt.close('all')