import os
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import itertools
import json
from B_SCR.general_functions import load_json_file, new_dir_add_dic

marker = itertools.cycle(('s', 'v', 'o', 'd', '+', '*', 'x'))
style.use('tableau-colorblind10')
plt.rcParams["figure.figsize"] = (6, 6)
##set font size
font = {'family': 'sans-serif',
		'weight': 'normal',
		'size': 14}
plt.rc('font', **font)


def experimental_results(material=None,
						 res_pth=None,
						 marker_dic=None):
	"""
	PLOT RAW EXPERIMENTAL DATA FOR FORCE-DISPLACEMENT FOR
	EACH MATERIAL ON SAME PLOT
	:param material: List of strings representing material
	:param res_pth: dictionary of paths to various files
	:param marker_dic: dictionary of markers and label data for materials
	:return: NOTHING (plot is saved to specific location)
	"""

	fig, ax2d = plt.subplots()
	ax = np.ravel(ax2d)
	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
	max_y = 0
	max_x = 0

	# ##FOR EACH MATERIAL GET THE PATH TO RESULTS AND PLOT DATA
	for m in material:
		# ##USING PATH DIC FIND GET RELEVANT DATA
		exp_df = pd.read_csv(res_pth['RAW_%s' % (m)], header=[0, 1])
		exp_df = exp_df.droplevel(level=1, axis=1)
		# ##GET MAX FORCE AND MAX DISPLACEMENT
		mf = exp_df['FORCE'].max()
		md = exp_df['DISPLACEMENT'].max()
		if mf > max_y:
			max_y = mf
		if md > max_x:
			max_x = md
		# ##PLOT DATA
		ax[0].plot(exp_df['DISPLACEMENT'],
				   exp_df['FORCE'],
				   linestyle='None',
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=0.05,
				   color='k',
				   label='Dataset %s' % (marker_dic[m]['number']))
		# ##PLOT DATA
		ax[0].plot(exp_df['DISPLACEMENT'],
				   exp_df['FORCE'],
				   linestyle='None',
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=[0, -1],
				   color='k')
	# # AXES LIMITS
	xlim_max = int(np.ceil(max_x))
	ylim_max = int(np.ceil(max_y))
	ax[0].set_xlim([0, xlim_max])
	ax[0].set_ylim([0, ylim_max])

	# ##AT LEAST FIVE TICK MARKS ON X AND Y AXES
	plt.xticks(np.linspace(0, xlim_max, 6))
	plt.yticks(np.linspace(0, ylim_max, 6))

	# ##AXES LABELS
	ax[0].set_xlabel('Displacement, mm')
	ax[0].set_ylabel('Force, kN')
	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(1, 0),
				 loc='lower right',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_on()
	### save figure
	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
							 'EXPERIMENTAL_RESULTS.png'),
				dpi=300,
				bbox_inches='tight')


def true_stress_strain(material=None,
					   res_pth=None,
					   marker_dic=None):
	"""
	PLOT TRUE STRESS-STRAIN FOR EACH MATERIAL ON SAME PLOT

	:param material: List of strings representing material
	:param res_pth: dictionary of paths to various files
	:param marker_dic: dictionary of markers and label data for materials
	:return: NOTHING (plot is saved to specific location)
	"""

	fig, ax2d = plt.subplots()
	ax = np.ravel(ax2d)
	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
	max_y = 0
	max_x = 0

	# ##FOR EACH MATERIAL GET THE PATH TO RESULTS AND PLOT DATA
	for m in material:
		# ##USING PATH DIC FIND GET RELEVANT DATA
		df = pd.read_csv(os.path.join(res_pth['RES_%s' % (m)], 'TRUE_INTERP.csv'), )
		# ##CONVERT STRAIN TO %
		df['TRUE_STRAIN'] = df['TRUE_STRAIN'] * 100
		# ##GET MAX FORCE AND MAX DISPLACEMENT
		mf = df['TRUE_STRESS'].max()
		md = df['TRUE_STRAIN'].max()
		if mf > max_y:
			max_y = mf
		if md > max_x:
			max_x = md
		# ##PLOT DATA
		ax[0].plot(df['TRUE_STRAIN'],
				   df['TRUE_STRESS'],
				   linestyle='None',
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=0.05,
				   color='k',
				   label='Dataset %s' % (marker_dic[m]['number']))
		# ##PLOT DATA
		ax[0].plot(df['TRUE_STRAIN'],
				   df['TRUE_STRESS'],
				   linestyle='None',
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=[0, -1],
				   color='k')

	# # AXES LIMITS
	xlim_max = int(np.ceil(max_x))
	ylim_max = int(np.ceil(max_y / 10) * 10)
	ax[0].set_xlim([0, xlim_max])
	ax[0].set_ylim([0, ylim_max])

	# AT LEAST FIVE TICK MARKS ON X AND Y AXES
	plt.xticks(np.linspace(0, xlim_max, 6))
	plt.yticks(np.linspace(0, ylim_max, 6))

	# AXES LABELS
	ax[0].set_xlabel('True strain, %')
	ax[0].set_ylabel('True stress, MPa')

	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(1, 0),
				 loc='lower right',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_on()
	### save figure
	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
							 'TRUE_STRESS_STRAIN.png'),
				dpi=300,
				bbox_inches='tight')


def second_derivative(material=None,
					  res_pth=None,
					  marker_dic=None):
	"""
	PLOT TRUE STRESS-STRAIN AND BEST SECOND DERIVATIVE  FOR EACH MATERIAL ON SAME PLOT
	NOTE: ONLY INTERESTED IN THE LINEAR REGIONS - IDENTIFICATION OF YIELD STRENGTH, YOUNG'S MODULUS

	:param material: List of strings representing material
	:param res_pth: dictionary of paths to various files
	:param marker_dic: dictionary of markers and label data for materials
	:return: NOTHING (plot is saved to specific location)
	"""

	fig, ax2d = plt.subplots()
	ax = np.ravel(ax2d)
	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
	max_y = 0
	max_x = 0

	# ##FOR EACH MATERIAL GET THE PATH TO RESULTS AND PLOT DATA
	for m in material:
		# ##USING PATH DIC FIND GET RELEVANT DATA
		df = pd.read_csv(os.path.join(res_pth['RES_%s' % (m)], 'TRUE_INTERP.csv'), )
		# ##CONVERT STRAIN TO %
		df['TRUE_STRAIN_PC'] = df['TRUE_STRAIN'] * 100
		# ##READ IN WINDOW LENGTH INFORMATION
		wl = pd.read_csv(os.path.join(res_pth['RES_%s' % (m)], 'WINDOW_LENGTH.csv'), index_col=0).reset_index(drop=True)
		# ##ORDER WL BY MULTI COLUMN AND GET 'BEST' .ILOC[0] RESULT
		wl.sort_values(by='MULTI', ascending=False)
		wl = wl.iloc[0]
		# ##LIMIT THE DF TO VALUES LESS THAN 10% GREATER THAN YIELD
		df = df[df['TRUE_STRAIN_PC'] <= 0.6]
		# ##FIND THE STRAIN/STRESS AT YIELD
		y_df = df[df['TRUE_STRESS'] == wl['SIGMA_Y']]
		# ##READ IN LINEAR DATA
		linear = pd.read_csv(
			os.path.join(res_pth['RES_%s' % (m)], 'LINEAR_REGION_WL%s.csv' % (str(int(wl['WINDOW_LENGTH'])))))
		# ##GET MAX FORCE AND MAX DISPLACEMENT
		mf = df['TRUE_STRESS'].max()
		md = df['TRUE_STRAIN_PC'].max()
		if mf > max_y:
			max_y = mf
		if md > max_x:
			max_x = md

		# ##PLOT DATA TRUE STRESS-STRAIN
		ax[0].plot(df['TRUE_STRAIN_PC'],
				   df['TRUE_STRESS'],
				   linestyle='None',
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=0.05,
				   color='k',
				   label='Dataset %s' % (marker_dic[m]['number']))
		# ##PLOT DATA
		ax[0].plot(df['TRUE_STRAIN_PC'],
				   df['TRUE_STRESS'],
				   linestyle='None',
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=[0, -1],
				   color='k')
		### ##PLOT LINEAR
		ax[0].plot(linear['TRUE_STRAIN'],
				   linear['PRED_TRUE_STRESS'],
				   label='Dataset %s linear fit' % (marker_dic[m]['number']))
		# ###PLOT THE YIELD STRENGTH
		ax[0].scatter(y_df['TRUE_STRAIN_PC'],
					  y_df['TRUE_STRESS'],
					  marker=marker_dic[m]['marker'],
					  color='b',
					  label='Dataset %s, $\sigma_y$: %s MPa' % (marker_dic[m]['number'], round(y_df['TRUE_STRESS'].values[0], 1)))

	# # AXES LIMITS
	xlim_max = max_x
	ylim_max = int(np.ceil(max_y / 10) * 10)
	ax[0].set_xlim([0, xlim_max])
	ax[0].set_ylim([0, ylim_max])

	# AT LEAST FIVE TICK MARKS ON X AND Y AXES
	plt.xticks(np.linspace(0, xlim_max, 6))
	plt.yticks(np.linspace(0, ylim_max, 6))

	# AXES LABELS
	ax[0].set_xlabel('True strain, %')
	ax[0].set_ylabel('True stress, MPa')

	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(1, 0),
				 loc='lower right',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_on()
	### save figure
	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
							 'SECOND_DERIV.png'),
				dpi=300,
				bbox_inches='tight')


def plastic_strain_extended(material=None,
							res_pth=None,
							marker_dic=None):
	"""
		PLOT TRUE STRESS- PLASTIC STRAIN FOR TWO POTENTIAL SLOPE VALUES

		:param material: List of strings representing material
		:param res_pth: dictionary of paths to various files
		:param marker_dic: dictionary of markers and label data for materials
		:return: NOTHING (plot is saved to specific location)
		"""
	fig, ax2d = plt.subplots()
	ax = np.ravel(ax2d)
	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
	max_y = 0
	max_x = 0

	# ##FOR EACH MATERIAL GET THE PATH TO RESULTS AND PLOT DATA
	for m in material:
		if m=='P91_20_2':

			# ##READ IN THE PROPERTIES OF THE MATERIAL
			dic = load_json_file(os.path.join(res_pth['RES_%s' % (m)], '%s_properties.txt' % (m)))
			slopes = dic['SLOPE'].keys()
			mymin, mymax = min([float(m) for m in slopes]), max([float(m) for m in slopes])

			# ##read in slope of zero
			mzero = pd.read_csv(dic['SLOPE'][str(mymin)]['ABAQUS_PLASTIC'])
			mzero['PLASTIC_STRAIN'] = mzero['PLASTIC_STRAIN'] * 100
			# ##READ IN LARGEST SLOPE
			mmax = pd.read_csv(dic['SLOPE'][str(mymax)]['ABAQUS_PLASTIC'])
			mmax['PLASTIC_STRAIN'] = mmax['PLASTIC_STRAIN'] * 100
			# ##READ IN THE EXPERIMENTAL UP TO UTS
			uts = pd.read_csv(os.path.join(res_pth['RES_%s' % (m)], 'ABA_TSPE_UTS.csv'))
			uts['PLASTIC_STRAIN'] = uts['PLASTIC_STRAIN'] * 100

			# ##LIMIT THE TWO DFS TO DATA ABOVE UTS ONLY & LIMIT PLASTIC STRAIN TO 20%
			mzero = mzero[(mzero['TRUE_STRESS'] >= uts['TRUE_STRESS'].iloc[-1]) & (mzero['PLASTIC_STRAIN']<=20)]
			mmax = mmax[(mmax['TRUE_STRESS'] >= uts['TRUE_STRESS'].iloc[-1])& (mmax['PLASTIC_STRAIN']<=20)]

			# ##GET MAX TRUE STRESS AND MAX PLASTIC STRAIN
			mf = max(mzero['TRUE_STRESS'].max(), mmax['TRUE_STRESS'].max())
			md = max(mzero['PLASTIC_STRAIN'].max(), mmax['PLASTIC_STRAIN'].max())
			if mf > max_y:
				max_y = mf
			if md > max_x:
				max_x = md

			# ##PLOT THE RANGE OF DATA FROM UTS INFO
			ax[0].plot(uts['PLASTIC_STRAIN'],
					   uts['TRUE_STRESS'],
					   linestyle='None',
					   marker=marker_dic[m]['marker'],
					   mfc='none',
					   markevery=50,
					   color='k',
					   label='Dataset %s' % (marker_dic[m]['number']))

			# ##PLOT DATA TRUE STRESS-PLASTIC STRAIN SLOPE 0
			ax[0].plot(mzero['PLASTIC_STRAIN'],
					   mzero['TRUE_STRESS'],
					   linestyle='None',
					   marker=marker_dic[m]['marker'],
					   mfc='none',
					   markevery=3,
					   label='Extended with slope: %s MPa' % (mymin))
			# ##PLOT DATA TRUE STRESS-PLASTIC STRAIN SLOPE 1
			ax[0].plot(mmax['PLASTIC_STRAIN'],
					   mmax['TRUE_STRESS'],
					   linestyle='None',
					   marker=marker_dic[m]['marker'],
					   mfc='none',
					   markevery=3,
					   label='Extended with slope: %s MPa' % (mymax))

	# # AXES LIMITS
	xlim_max = int(np.ceil(max_x / 10) * 10)
	ylim_max = int(np.ceil(max_y / 100) * 100)
	ax[0].set_xlim([0, xlim_max])
	ax[0].set_ylim([0, ylim_max])

	# AT LEAST FIVE TICK MARKS ON X AND Y AXES
	plt.xticks(np.linspace(0, xlim_max, 6))
	plt.yticks(np.linspace(0, ylim_max, 6))

	# AXES LABELS
	ax[0].set_xlabel('Plastic strain, %')
	ax[0].set_ylabel('True stress, MPa')

	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(1, 0),
				 loc='lower right',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_on()
	### save figure
	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
							 'ABAQUS_SLOPE_COMPARISON.png'),
				dpi=300,
				bbox_inches='tight')
#
# def general_eng_ss(res_pth=None,
# 				   marker_dic=None):
# 	"""
# 	This creates a basic diagram of engineering stress-strain
# 	for CS graduates who may not understand material behaviour
# 	:return:
# 	"""
#
# 	fig, ax2d = plt.subplots()
# 	ax = np.ravel(ax2d)
# 	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
# 	max_y = 0
# 	max_x = 0
#
# 	# ##READ IN THE PROPERTIES OF THE MATERIAL
# 	dic = {'SIGMA_Y':200,
# 		   'EPS_Y':0.037,
# 		   'UTS': 301,
# 		   'EPS_UTS':0.25,
# 		   'EPS_F':0.435}
# 	# ##READ IN DATA
# 	exp_df = pd.read_csv(os.path.join(os.path.join(res_pth['cwd'], 'OUTPUT'), 'GENERAL_SS.csv'))
# 	exp_df = exp_df.iloc[0:-1]
# 	# xnew = np.linspace(exp_df['STRAIN'].iloc[0], exp_df['STRAIN'].iloc[-1], num=len(exp_df) * 4)
# 	# f = interp1d(exp_df['STRAIN'], exp_df['STRESS'], kind='quadratic')
# 	# extrap = pd.DataFrame()
# 	# extrap['STRAIN'] = xnew
# 	# extrap['STRESS'] = f(xnew)
# 	#
# 	# ##LINEAR INTERPOLATION
# 	a = exp_df[(exp_df['STRESS']<=dic['SIGMA_Y']) & (exp_df['STRAIN']<=dic['EPS_Y'])]
# 	xnew = np.linspace(a['STRAIN'].iloc[0], a['STRAIN'].iloc[-1], num=len(exp_df) * 4)
# 	f = interp1d(a['STRAIN'], a['STRESS'], kind='linear')
# 	linear = pd.DataFrame()
# 	linear['STRAIN'] = xnew
# 	linear['STRESS'] = f(xnew)
# 	# ##HARDENING INTERPOLATION
# 	a = exp_df[(exp_df['STRAIN']>=dic['EPS_Y']) & (exp_df['STRAIN']<=dic['EPS_UTS'])]
# 	f = interp1d(a['STRAIN'], a['STRESS'], kind='quadratic')
# 	xnew = np.linspace(a['STRAIN'].iloc[0], a['STRAIN'].iloc[-1], num=len(a) * 5)
# 	hardening = pd.DataFrame()
# 	hardening['STRAIN'] = xnew
# 	hardening['STRESS'] = f(xnew)
# 	# ##SOFTENING INTERPOLATION
# 	a = exp_df[(exp_df['STRAIN'] >= dic['EPS_UTS'])]
# 	f = interp1d(a['STRAIN'], a['STRESS'], kind='quadratic')
# 	xnew = np.linspace(a['STRAIN'].iloc[0], a['STRAIN'].iloc[-1], num=len(a) * 10)
# 	soften = pd.DataFrame()
# 	soften['STRAIN'] = xnew
# 	soften['STRESS'] = f(xnew)
#
# 	# ##RECOMBINE
# 	extrap = pd.DataFrame()
# 	extrap = pd.concat([linear, hardening], axis=0, ignore_index=True)
# 	extrap = pd.concat([extrap, soften], axis=0, ignore_index=True)
# 	extrap = extrap.drop_duplicates(keep='first')
#
# 	# ##GET MAX FORCE AND MAX DISPLACEMENT
# 	mf = extrap['STRESS'].max()
# 	md = extrap['STRAIN'].max()
# 	if mf > max_y:
# 		max_y = mf * 1.05
# 	if md > max_x:
# 		max_x = md * 1.05
# 	# ##PLOT DATA
# 	ax[0].plot(extrap['STRAIN'],
# 			   extrap['STRESS'],
# 			   mfc='none',
# 			   markevery=0.05,
# 			   color='k',
# 			   label='Data')
# 	# ##PLOT DATA
# 	ax[0].plot(extrap['STRAIN'],
# 			   extrap['STRESS'],
# 			   mfc='none',
# 			   markevery=[0, -1],
# 			   color='k')
#
# 	# ##ADD ANNOTATIONS
# 	# ##YOUNGS MODULUS
# 	byield = extrap[extrap['STRESS']<=dic['SIGMA_Y']]
# 	# ##LIMIT THE DF TO REGION WHERE YOU WANT THE LINES TO APPEAR
# 	byield = byield[(byield['STRESS']>=100) & (byield['STRESS']<=200)].reset_index(drop=True)
# 	# ##VLINE, YOUNGS
# 	ax[0].vlines(x=byield.iloc[-1]['STRAIN'],
# 				  ymin=byield.iloc[0]['STRESS'],
# 				  ymax=byield.iloc[-1]['STRESS'])
# 	# ##HLINE, YOUNGS
# 	ax[0].hlines(y=byield.iloc[0]['STRESS'],
# 				  xmax=byield.iloc[0]['STRAIN'],
# 				  xmin=byield.iloc[-1]['STRAIN'])
# 	ax[0].annotate('1', xy=(byield['STRAIN'].iloc[int(len(byield)/2)], byield['STRESS'].iloc[0]*0.85), horizontalalignment='center')
# 	ax[0].annotate('E', xy=(byield['STRAIN'].iloc[-1]*1.2, byield['STRESS'].iloc[int(len(byield)/2)]), horizontalalignment='center', verticalalignment='center')
#
# 	# ##LINEAR REGION
# 	sh = extrap[(extrap['STRESS'] <= dic['SIGMA_Y']) & (extrap['STRAIN'] <= dic['EPS_UTS'])]
# 	ax[0].plot(sh['STRAIN'],
# 			   sh['STRESS'],
# 			   color='green',
# 			   linewidth=10,
# 			   alpha=0.4,
# 			  zorder=0,
# 			   label='Linear region')
#
# 	# ##STRAIN HARDENING REGION
# 	sh = extrap[(extrap['STRESS'] >= dic['SIGMA_Y']) & (extrap['STRESS'] <= dic['UTS']) &
# 				(extrap['STRAIN'] <= dic['EPS_UTS'])]
# 	ax[0].plot(sh['STRAIN'],
# 			   sh['STRESS'],
# 			   color='orange',
# 			   linewidth=10,
# 		   		alpha=0.4,
# 				zorder=0,
# 			   label='Strain hardening region')
#
# 	# ##DAMAGE REGION
# 	damage = extrap[(extrap['STRAIN'] >= dic['EPS_UTS'])]
# 	ax[0].plot(damage['STRAIN'],
# 			   damage['STRESS'],
# 			   color='indigo',
# 			   linewidth=10,
# 			   alpha=0.4,
# 			   zorder=0,
# 			   label='Material damage region')
#
# 	# ## PLOT YIELD POSITION
# 	ax[0].scatter(dic['EPS_Y'],
# 				  dic['SIGMA_Y'],
# 				  marker='o',
# 				  color='b',
# 				  s=75,
# 				  zorder=1)
# 	# ###HORIZONTAL SIGMA
# 	ax[0].hlines(y=dic['SIGMA_Y'],
# 				 xmax=dic['EPS_Y'],
# 				 xmin=0,
# 				 linestyle='--',
# 				 color='k')
# 	ax[0].annotate('$\sigma_y$', xy=(dic['EPS_Y']/2, dic['SIGMA_Y']*1.03),
# 				   horizontalalignment='center',
# 				   verticalalignment='center')
#
#
# 	# ##PLOT UTS POSITION
# 	ax[0].scatter(dic['EPS_UTS'],
# 				  dic['UTS'],
# 				  marker='^',
# 				  color='b',
# 				  s=75,
# 				  zorder=1,)
# 	# ###HORIZONTAL SIGMA
# 	ax[0].hlines(y=dic['UTS'],
# 				 xmax=dic['EPS_UTS'],
# 				 xmin=0,
# 				 linestyle='--',
# 				 color='k')
# 	ax[0].annotate('$\sigma_{UTS}$', xy=(dic['EPS_UTS'] / 2, dic['UTS'] * 1.03),
# 				   horizontalalignment='center',
# 				   verticalalignment='center')
#
# 	# ##AXES LABELS
# 	ax[0].set_xlabel('Engineering strain, $\epsilon_{eng}$')
# 	ax[0].set_ylabel('Engineering stress, $\sigma_{eng}$')
# 	# ##AXES LIMITS
# 	ax[0].set_xlim([0, max_x])
# 	ax[0].set_ylim([0, max_y])
# 	# ##AXES TICKS(EMPTY)
# 	ax[0].set_xticks([])
# 	ax[0].set_yticks([])
#
# 	# ##LEGEND
# 	ax[0].legend(bbox_to_anchor=(0.95, 0),
# 				 loc='lower right',
# 				 ncol=1,
# 				 borderaxespad=0,
# 				 frameon=False)
# 	# ##TURN ON MINOR TICKS
# 	plt.minorticks_off()
# 	# ## save figure
# 	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
# 							 'DIAGRAM_ENG_SS.png'),
# 				dpi=300,
# 				bbox_inches='tight')
#
#
# def general_true_ss(res_pth=None,
# 				   marker_dic=None):
# 	"""
# 	This creates a basic diagram of engineering stress-strain
# 	for CS graduates who may not understand material behaviour
# 	:return:
# 	"""
#
# 	fig, ax2d = plt.subplots()
# 	ax = np.ravel(ax2d)
# 	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
# 	max_y = 0
# 	max_x = 0
#
# 	# ##READ IN THE PROPERTIES OF THE MATERIAL
# 	dic = {'SIGMA_Y':200,
# 		   'EPS_Y':0.037,
# 		   'UTS': 301,
# 		   'EPS_UTS':0.25,
# 		   'EPS_F':0.435}
# 	# ##READ IN DATA
# 	exp_df = pd.read_csv(os.path.join(os.path.join(res_pth['cwd'], 'OUTPUT'), 'GENERAL_SS.csv'))
# 	# ##GET MAX FORCE AND MAX DISPLACEMENT
# 	mf = exp_df['STRESS'].max()
# 	md = exp_df['STRAIN'].max()
# 	if mf > max_y:
# 		max_y = mf * 1.05
# 	if md > max_x:
# 		max_x = md * 1.05
# 	# ##PLOT DATA
# 	ax[0].plot(exp_df['STRAIN'].iloc[0:-2],
# 			   exp_df['STRESS'].iloc[0:-2],
# 			   mfc='none',
# 			   markevery=0.05,
# 			   color='k',
# 			   label='With damage')
# 	# ##PLOT DATA
# 	ax[0].plot(exp_df['TRUE_STRAIN'],
# 			   exp_df['TRUE_STRESS'],
# 			   color='k',
# 			   linestyle='--',
# 			   label='No damage')
#
#
# 	# ##AXES LABELS
# 	ax[0].set_xlabel('True strain')
# 	ax[0].set_ylabel('True stress')
# 	# ##AXES LIMITS
# 	ax[0].set_xlim([0, max_x])
# 	ax[0].set_ylim([0, max_y])
# 	# ##AXES TICKS(EMPTY)
# 	ax[0].set_xticks([])
# 	ax[0].set_yticks([])
#
# 	# ##LEGEND
# 	ax[0].legend(bbox_to_anchor=(0.95, 0),
# 				 loc='lower right',
# 				 ncol=1,
# 				 borderaxespad=0,
# 				 frameon=False)
# 	# ##TURN ON MINOR TICKS
# 	plt.minorticks_off()
# 	# ## save figure
# 	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
# 							 'DIAGRAM_TRUE_SS.png'),
# 				dpi=300,
# 				bbox_inches='tight')


def general_eng_ss(material=None,
				   res_pth=None,
				   marker_dic=None):
	"""
	PLOT RAW EXPERIMENTAL DATA FOR FORCE-DISPLACEMENT FOR
	EACH MATERIAL ON SAME PLOT
	:param material: List of strings representing material
	:param res_pth: dictionary of paths to various files
	:param marker_dic: dictionary of markers and label data for materials
	:return: NOTHING (plot is saved to specific location)
	"""

	fig, ax2d = plt.subplots()
	ax = np.ravel(ax2d)
	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
	max_y = 0
	max_x = 0

	# ##READ IN DATA
	exp_df = pd.read_csv(os.path.join(res_pth, 'DF_SS.csv'))
	# ##NEED TO SHIFT THE STRAIN OVER 10% FOR VISIBILITY
	exp_df['SHIFT'] = exp_df['STRAIN'] * 10
	# ##READ IN PROPERTIES
	dic = load_json_file(os.path.join(res_pth, '%s_properties.txt'%(material)))

	# ##GET MAX FORCE AND MAX DISPLACEMENT
	mf = exp_df['STRESS'].max()
	md = exp_df['STRAIN'].max()
	# ##GET LOCATION FOR UTS
	mloc = exp_df.iloc[(exp_df['STRESS']-dic['ENG_STRESS']).abs().argsort()[:1]].index[0]
	# ##GET LOCATION FOR YIELD
	lim = exp_df[exp_df['STRAIN']<=dic['ENG_STRAIN']]
	yloc = lim.iloc[(lim['STRESS']-dic['SIGMA_Y']).abs().argsort()[:1]].index[0]
	if mf > max_y:
		max_y = mf * 1.05
	if md > max_x:
		max_x = md * 1.05
	# ##PLOT DATA
	ax[0].plot(exp_df['STRAIN'],
			   exp_df['STRESS'],
			   mfc='none',
			   markevery=0.05,
			   color='k',
			   label='Data')
	# ##PLOT DATA
	ax[0].plot(exp_df['STRAIN'],
			   exp_df['STRESS'],
			   mfc='none',
			   markevery=[0, -1],
			   color='k')

	# ##ADD ANNOTATIONS
	# ##YOUNGS MODULUS
	byield = exp_df[(exp_df['STRESS']>=(dic['SIGMA_Y']/2)) &
					(exp_df['STRESS']<=dic['SIGMA_Y']*1.1) &
					(exp_df.index<=yloc)]
	# ##VLINE, YOUNGS
	ax[0].vlines(x=byield.iloc[-1]['SHIFT'],
				  ymin=byield.iloc[0]['STRESS'],
				  ymax=byield.iloc[-1]['STRESS'])
	# ##HLINE, YOUNGS
	ax[0].hlines(y=byield.iloc[0]['STRESS'],
				  xmax=byield.iloc[-1]['SHIFT'],
				  xmin=byield.iloc[-1]['STRAIN'])
	ax[0].annotate('1', xy=(byield['SHIFT'].iloc[int(len(byield)/2)],
							byield['STRESS'].iloc[0]*0.85),
				   horizontalalignment='center')
	ax[0].annotate('E', xy=(byield['SHIFT'].iloc[-1]*1.2,
							byield['STRESS'].iloc[2]),
				   horizontalalignment='center', verticalalignment='center')

	# ##LINEAR REGION
	sh = exp_df[(exp_df['STRESS'] <= dic['SIGMA_Y']) & (exp_df['STRAIN'] <= dic['ENG_STRAIN'])]
	ax[0].plot(sh['STRAIN'],
			   sh['STRESS'],
			   color='green',
			   linewidth=10,
			   alpha=0.4,
			  zorder=0,
			   label='Linear region')

	# ##STRAIN HARDENING REGION
	sh = exp_df[(exp_df['STRESS'] >= dic['SIGMA_Y']) &
				(exp_df['STRESS'] <= dic['ENG_STRESS']) &
				(exp_df['STRAIN'] <= dic['ENG_STRAIN'])]
	ax[0].plot(sh['STRAIN'],
			   sh['STRESS'],
			   color='orange',
			   linewidth=10,
		   		alpha=0.4,
				zorder=0,
			   label='Strain hardening region')

	# ##DAMAGE REGION
	damage = exp_df[(exp_df['STRAIN'] >= dic['ENG_STRAIN'])]
	ax[0].plot(damage['STRAIN'],
			   damage['STRESS'],
			   color='indigo',
			   linewidth=10,
			   alpha=0.4,
			   zorder=0,
			   label='Material damage region')

	# ## PLOT YIELD POSITION
	ax[0].scatter(exp_df['STRAIN'].iloc[yloc],
				  exp_df['STRESS'].iloc[yloc],
				  marker='o',
				  color='b',
				  s=75,
				  zorder=1)
	# ###HORIZONTAL SIGMA
	ax[0].hlines(y=exp_df['STRESS'].iloc[yloc],
				 xmax=exp_df['SHIFT'].iloc[yloc],
				 xmin=0,
				 linestyle='--',
				 color='k')
	ax[0].annotate('$\sigma_y$',
				   xy=(exp_df['SHIFT'].iloc[yloc]/2, exp_df['STRESS'].iloc[yloc]*1.03),
				   horizontalalignment='center',
				   verticalalignment='center')


	# ##PLOT UTS POSITION
	ax[0].scatter(dic['ENG_STRAIN'],
				  dic['ENG_STRESS'],
				  marker='^',
				  color='b',
				  s=75,
				  zorder=1,)
	# ###HORIZONTAL SIGMA
	ax[0].hlines(y=dic['ENG_STRESS'],
				 xmax=dic['ENG_STRAIN'],
				 xmin=0,
				 linestyle='--',
				 color='k')
	ax[0].annotate('$\sigma_{UTS}$', xy=(dic['ENG_STRAIN'] / 2, dic['ENG_STRESS'] * 1.03),
				   horizontalalignment='center',
				   verticalalignment='center')

	# ##AXES LABELS
	ax[0].set_xlabel('Engineering strain, $\epsilon_{eng}$')
	ax[0].set_ylabel('Engineering stress, $\sigma_{eng}$')
	# ##AXES LIMITS
	ax[0].set_xlim([0, max_x])
	ax[0].set_ylim([0, max_y])
	# ##AXES TICKS(EMPTY)
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(0.95, 0),
				 loc='lower right',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_off()
	# ## save figure
	pth = os.path.join(res_pth, '..')
	plt.savefig(os.path.join(pth,
							 'PAPERA_PLOTS\\DIAGRAM_ENG_SS.png'),
				dpi=300,
				bbox_inches='tight')


def general_true_ss(material=None,
				   res_pth=None,
				   marker_dic=None):
	"""
	This creates a basic diagram of engineering stress-strain
	for CS graduates who may not understand material behaviour
	:return:
	"""

	fig, ax2d = plt.subplots()
	ax = np.ravel(ax2d)
	# ##INITIAL MAXIMUM AS ZERO THESE WILL BE ITERATED TO BE THE GREATEST X,Y IN ANY OF THE RESULTS
	max_y = 0
	max_x = 0

	# ##READ IN DATA
	exp_df = pd.read_csv(os.path.join(res_pth, 'DF_SS.csv'))
	true_df = exp_df.dropna()
	# ##READ IN PROPERTIES
	dic = load_json_file(os.path.join(res_pth, '%s_properties.txt' % (material)))
	m = 600
	x1 = true_df['TRUE_STRAIN'].iloc[-1]
	y1 = true_df['TRUE_STRESS'].iloc[-1]
	x2 = exp_df['STRAIN'].iloc[-1]
	## calc intercept c
	c = y1 - m*x1
	# ##CREATE ARRAY OF STRAINS TO EXTRAP
	ext_strain = np.arange(x1, x2, 1e-3)
	# ##CREATE STRESSES FOR THE EXTRAPOLATED STRAINS
	ext_stress = m * ext_strain + c

	# ##GET MAX FORCE AND MAX DISPLACEMENT
	mf = max(exp_df['TRUE_STRESS'].max(), ext_stress.max())
	md = exp_df['STRAIN'].max()
	if mf > max_y:
		max_y = mf * 1.05
	if md > max_x:
		max_x = md * 1.05
	# ##PLOT DATA
	ax[0].plot(exp_df['STRAIN'].iloc[0:-2],
			   exp_df['STRESS'].iloc[0:-2],
			   mfc='none',
			   markevery=0.05,
			   color='k',
			   label='With damage')
	# ##PLOT TRUE DATA
	ax[0].plot(exp_df['TRUE_STRAIN'],
			   exp_df['TRUE_STRESS'],
			   color='k',
			   linestyle='--',
			   label='No damage')
	# ##PLOT EXTRAPOLATED TRUE
	ax[0].plot(ext_strain,
			   ext_stress,
			   color='k',
			   linestyle='--')


	# ##AXES LABELS
	ax[0].set_xlabel('True strain')
	ax[0].set_ylabel('True stress')
	# ##AXES LIMITS
	ax[0].set_xlim([0, max_x])
	ax[0].set_ylim([0, max_y])
	# ##AXES TICKS(EMPTY)
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(0.95, 0),
				 loc='lower right',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_off()
	# ## save figure
	pth = os.path.join(res_pth, '..')
	plt.savefig(os.path.join(pth,
							 'PAPERA_PLOTS\\DIAGRAM_TRUE_SS.png'),
				dpi=300,
				bbox_inches='tight')


####################################################################
# ####MATERIALS
materials = ['P91_20_1', 'P91_20_2', 'P91_500']
# ##EMPTY DICTIONARY FOR PATHS
pdic = {}
# ##DEFINE PROJECT DIRECTORY
pdic['cwd'] = os.getcwd()
# ##FOR DIRECTORY IN PROJECT GET DIRECTORIES AND PATH TO DIRECTORY
for r, d, f in os.walk(pdic['cwd']):
	if 'OUTPUT' in r:
		for material in materials:
			pdic['RES_%s' % (material)] = os.path.join(pdic['cwd'], 'OUTPUT\\%s') % (material)
			pdic['RAW_%s' % (material)] = os.path.join(pdic['cwd'], 'A_RAW_DATA\\%s.csv' % (material))
# ##CREATE DIRECTORY FOR PAPER_PLOTS
pdic = new_dir_add_dic(dic=pdic,
					   key='PAPERA_PLOTS',
					   path=os.path.join(pdic['cwd'], 'OUTPUT'),
					   dir_name='PAPERA_PLOTS',
					   exist_ok=True)
# ##SPECIFY SYMBOLS AND NUMBERS FOR DATASETS
marker_dic = {}
# ##CREATE DIRECTORY FOR INDIVIDUAL MATERIALS IN PAPER PLOTS
for material in materials:
	pdic = new_dir_add_dic(dic=pdic,
						   key='PLOTS_%s' % (material),
						   path=pdic['PAPERA_PLOTS'],
						   dir_name=material,
						   exist_ok=True)
	if material == 'P91_20_1':
		mynum = 2
		mm = next(marker)
	elif material == 'P91_20_2':
		mynum = 1
		mm = next(marker)
	elif material == 'P91_500':
		mynum = 3
		mm = next(marker)
	# ##SET THE SYMBOLS AND LABEL
	marker_dic[material] = {'marker': mm, 'number': mynum}
# ##SORT MARKER DICTIONARY BY NUMBER RETURN LIST OF MATERIALS IN ORDER OF NUMBER
order = sorted(marker_dic, key=lambda x: marker_dic[x]['number'])
# # # ##PLOT GENERAL ILLUSTRATIVE DIAGRAMS
# general_eng_ss(material=order[0],
# 				  res_pth=os.path.join(os.getcwd(), 'OUTPUT\\P91_20_2'),
# 				  marker_dic=marker_dic)
# general_true_ss(material=order[0],
# 				  res_pth=os.path.join(os.getcwd(), 'OUTPUT\\P91_20_2'),
# 				  marker_dic=marker_dic)
# # ## PLOT ALL EXPERIMENTAL FORCE-DISPLACEMENT
# experimental_results(material=order,
# 					 res_pth=pdic,
# 					 marker_dic=marker_dic)
# # ##PLOT ALL TRUE STRESS-STRAIN
# true_stress_strain(material=order,
# 				   res_pth=pdic,
# 				   marker_dic=marker_dic)
# second_derivative(material=order[0],
# 				  res_pth=os.path.join(os.getcwd(), 'OUTPUT/P91_20_2'),
# 				  marker_dic=marker_dic)
plastic_strain_extended(material=order,
						res_pth=pdic,
						marker_dic=marker_dic)
