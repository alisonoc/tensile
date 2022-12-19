import os
import numpy as np
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
					  label='Dataset %s, $\sigma_y$: %s' % (marker_dic[m]['number'], round(y_df['TRUE_STRESS'].values[0], 1)))

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

			# ##LIMIT THE TWO DFS TO DATA ABOVE UTS ONLY
			mzero = mzero[mzero['TRUE_STRESS'] >= uts['TRUE_STRESS'].iloc[-1]]
			mmax = mmax[mmax['TRUE_STRESS'] >= uts['TRUE_STRESS'].iloc[-1]]

			# ##GET MAX FORCE AND MAX DISPLACEMENT
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
					   markevery=0.05,
					   label='Extended with slope: %s' % (mymin))
			# ##PLOT DATA TRUE STRESS-PLASTIC STRAIN SLOPE 1
			ax[0].plot(mmax['PLASTIC_STRAIN'],
					   mmax['TRUE_STRESS'],
					   linestyle='None',
					   marker=marker_dic[m]['marker'],
					   mfc='none',
					   markevery=0.05,
					   label='Extended with slope: %s' % (mymax))

	# # AXES LIMITS
	xlim_max = int(np.ceil(max_x / 100) * 100)
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
	ax[0].legend(bbox_to_anchor=(0.00, 1),
				 loc='upper left',
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

def general_eng_ss():
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

	# ##FOR EACH MATERIAL GET THE PATH TO RESULTS AND PLOT DATA
	for m in material:
		# ##READ IN THE PROPERTIES OF THE MATERIAL
		dic = load_json_file(os.path.join(res_pth['RES_%s' % (m)], '%s_properties.txt' % (m)))
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
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=0.05,
				   color='k',
				   label='Dataset %s' % (marker_dic[m]['number']))
		# ##PLOT DATA
		ax[0].plot(exp_df['DISPLACEMENT'],
				   exp_df['FORCE'],
				   marker=marker_dic[m]['marker'],
				   mfc='none',
				   markevery=[0, -1],
				   color='k')
		# ## PLOT YIELD POSITION
		my_yield = exp_df[exp_df['']]
		ax[0].scatter(dic['SIGMA_YIELD'])

	# AXES LABELS
	ax[0].set_xlabel('Plastic strain, %')
	ax[0].set_ylabel('True stress, MPa')
	# ##AXES TICKS(EMPTY)
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	# ##LEGEND
	ax[0].legend(bbox_to_anchor=(0.00, 1),
				 loc='upper left',
				 ncol=1,
				 borderaxespad=0,
				 frameon=False)
	# ##TURN ON MINOR TICKS
	plt.minorticks_on()
	### save figure
	plt.savefig(os.path.join(res_pth['PAPERA_PLOTS'],
							 'DIAGRAM_ENG_SS.png'),
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
			pdic['RES_%s' % (material)] = os.path.join(pdic['cwd'], 'OUTPUT/%s') % (material)
			pdic['RAW_%s' % (material)] = os.path.join(pdic['cwd'], 'A_RAW_DATA/%s.csv' % (material))
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
# ## PLOT ALL EXPERIMENTAL FORCE-DISPLACEMENT
experimental_results(material=order,
					 res_pth=pdic,
					 marker_dic=marker_dic)
# ##PLOT ALL TRUE STRESS-STRAIN
true_stress_strain(material=order,
				   res_pth=pdic,
				   marker_dic=marker_dic)
second_derivative(material=order,
				  res_pth=pdic,
				  marker_dic=marker_dic)
plastic_strain_extended(material=order,
						res_pth=pdic,
						marker_dic=marker_dic)
