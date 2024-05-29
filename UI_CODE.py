import sys
import numpy as np
import pandas as pd
import os
import copy
from pathlib import Path
from B_SCR.general_functions import new_dir_add_dic, merge_dicts, write_json_file
from B_SCR.material_properties import convert_fvd_engss, true_stress_strain, aoc_calc_slope
from B_SCR.plots import true_stress_true_strain, compare_interp_true, plot_sec_der_peaks, plot_linear, true_stress_plastic_strain, plot_all_slopes
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QPlainTextEdit, QHBoxLayout, QMenuBar, QAction, QMessageBox
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, max_error
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import shutil
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QLabel

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filePath = None
        self.plotPaths = []  # List to hold paths of generated plots
        self.dataPaths = []
        self.currentPlotIndex = 0
        self.initUI()
        self.setupDirectories(clear=True)
        self.setupGeometry()

    def initUI(self):
        self.setWindowTitle("Tensile Analysis")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout(self.centralWidget)

        # Setup button layout
        buttonLayout = QHBoxLayout()
        self.button = QPushButton('Open CSV', self)
        self.button.clicked.connect(self.openFileDialog)
        buttonLayout.addWidget(self.button)

        self.analyzeButton = QPushButton("Analyze", self)
        self.analyzeButton.clicked.connect(self.analyzeData)
        self.analyzeButton.setEnabled(False)  # Disabled at initialization
        buttonLayout.addWidget(self.analyzeButton)

        self.clearButton = QPushButton("Clear", self)
        self.clearButton.clicked.connect(self.clearContents)
        buttonLayout.addWidget(self.clearButton)

        self.quitButton = QPushButton("Quit", self)
        self.quitButton.clicked.connect(self.close)
        buttonLayout.addWidget(self.quitButton)

        layout.addLayout(buttonLayout)

        self.statusLabel = QLabel("Status: Waiting for data...")
        layout.addWidget(self.statusLabel)

        self.textArea = QPlainTextEdit(self)
        self.textArea.setReadOnly(True)
        layout.addWidget(self.textArea)

        self.prevButton = QPushButton("Previous", self)
        self.prevButton.clicked.connect(self.prevPlot)
        self.prevButton.setEnabled(False)
        buttonLayout.addWidget(self.prevButton)

        self.nextButton = QPushButton("Next", self)
        self.nextButton.clicked.connect(self.nextPlot)
        self.nextButton.setEnabled(False)
        buttonLayout.addWidget(self.nextButton)

        self.saveOutputsButton = QPushButton('Save Outputs', self)
        self.saveOutputsButton.clicked.connect(self.savePlot)
        self.saveOutputsButton.setEnabled(False)
        buttonLayout.addWidget(self.saveOutputsButton)

        self.saveAllButton = QPushButton("Save All Outputs", self)
        self.saveAllButton.clicked.connect(self.saveAllPlots)
        self.saveAllButton.setEnabled(False)
        buttonLayout.addWidget(self.saveAllButton)

        # Setup figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.loadingLabel = QLabel(self.centralWidget)
        self.movie = QMovie('../Spinner.gif')
        self.loadingLabel.setMovie(self.movie)
        layout.addWidget(self.loadingLabel)
        self.loadingLabel.hide()

    def startLoadingAnimation(self):
        self.loadingLabel.show()
        self.movie.start()

    def stopLoadingAnimation(self):
        self.movie.stop()
        self.loadingLabel.hide()

    def displayPlot(self):
        if self.plotPaths and 0 <= self.currentPlotIndex < len(self.plotPaths):
            plot_path = self.plotPaths[self.currentPlotIndex]
            self.figure.clear()  # Clear the existing plot
            ax = self.figure.add_subplot(111)
            img = mpimg.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            title = os.path.splitext(os.path.basename(plot_path))[0]  # Get filename without extension
            ax.set_title(title)
            self.canvas.draw()  # Refresh the canvas

    def prevPlot(self):
        if self.currentPlotIndex > 0:
            self.currentPlotIndex -= 1
            self.displayPlot()
            self.nextButton.setEnabled(True)  # Ensure the 'Next' button is enabled when moving back from the last plot

        # Disable 'Previous' button if this is the first plot
        if self.currentPlotIndex == 0:
            self.prevButton.setEnabled(False)

    def nextPlot(self):
        if self.currentPlotIndex < len(self.plotPaths) - 1:
            self.currentPlotIndex += 1
            self.displayPlot()
            self.prevButton.setEnabled(True)  # Ensure the 'Previous' button is enabled when moving forward from the first plot

        # Disable 'Next' button if this is the last plot
        if self.currentPlotIndex == len(self.plotPaths) - 1:
            self.nextButton.setEnabled(False)

    def savePlot(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","PNG Files (*.png);;All Files (*)", options=options)
        if fileName:
            shutil.copy(self.plotPaths[self.currentPlotIndex], fileName)


    def setupDirectories(self, clear=False):
        # Setup default paths
        self.path_dic = {
            'cwd': os.getcwd(),
            'project_dir': os.getcwd(),
            'output_dir': os.path.join(os.getcwd(), 'OUTPUT')
        }
         
        # Create OUTPUT directory if it doesn't exist
        self.path_dic = new_dir_add_dic(
            dic=self.path_dic,
            key='output_dir',
            path=self.path_dic['cwd'],
            dir_name='OUTPUT',
            exist_ok=True
        )
        if clear:
            shutil.rmtree(self.path_dic['output_dir'])
            os.makedirs(self.path_dic['output_dir'])


    def setupGeometry(self):
        self.geom_dic = {
            'GAUGE_LENGTH': 25.,
            'GAUGE_DIAMETER': 4.,
            'CONN_DIAMETER': 5.,
            'SPECIMEN_LENGTH': 72.,
            'ROUND_RADIUS': 3.,
            'THREADED_LENGTH': 15
        }
    
    def openFileDialog(self):
        #Open dialog and select CSV file
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files(*.csv)", options=options)
        if filePath:
            self.path_dic['exp_fvd'] = filePath
            print(f"File Selected: {filePath}")
            self.analyzeButton.setEnabled(True)  # Disabled at initialization
            self.loadCsv(filePath)
        else:
            self.statusLabel.setText("Status: No file selected")
            self.analyzeButton.setEnabled(False)

    def loadCsv(self, filePath):
        try:
            df = pd.read_csv(filePath, header=[0, 1]).droplevel(level=1, axis=1)
            self.textArea.setPlainText(str(df.head()))  # Ensure you convert DataFrame to string
            self.statusLabel.setText(f"Status: Loaded data from {filePath}")
            # self.path_dic['exp_fvd'] = filePath
            self.processFileAndDirectory(filePath)
        except Exception as e:
            print(f"Error: {str(e)}")  # Print more specific error message
            self.statusLabel.setText(f"Status: Failed to load data - {str(e)}")

    def processFileAndDirectory(self, filePath):
        # filename = os.path.basename(filePath)
        filePath = Path(filePath)
        material = filePath.stem
        
        # ##CREATE MATERIALS SUBDIRECTORY
        self.path_dic = new_dir_add_dic(
            dic=self.path_dic,
            key='curr_results',
            path=self.path_dic['output_dir'],
            dir_name=material,
            exist_ok=True
        )

    def saveAllPlots(self):
        # Ask the user to select a directory to save the plots
        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Save All Plots")
        if folder:  # Check if a folder was selected
            try:
                for plot_path in self.plotPaths:
                    shutil.copy(plot_path, os.path.join(folder, os.path.basename(plot_path)))
                self.statusLabel.setText("All plots have been successfully saved to: " + folder)
                QMessageBox.information(self, "Save Successful", "All plots have been saved successfully to:\n" + folder)
            except Exception as e:
                self.statusLabel.setText(f"Error saving plots: {str(e)}")
                QMessageBox.critical(self, "Save Error", f"An error occurred while saving the plots: {str(e)}")
        else:
            # If no directory is selected, notify the user
            self.statusLabel.setText("Save operation cancelled.")
            QMessageBox.warning(self, "Save Cancelled", "No directory selected; save operation cancelled.")

    #################################
    ## MATERIAL ASSESSMENT
    ################################
    def analyzeData(self):
        if self.path_dic['exp_fvd']:
            try:
                self.startLoadingAnimation()
                print(os.path.join(self.path_dic['curr_results']))
                output_dir = os.path.join(self.path_dic['curr_results'])
                fvd = pd.read_csv(self.path_dic['exp_fvd'], header=[0, 1]).droplevel(level=1, axis=1)
                # ##CALCULATE ENGINEERING STRESS-STRAIN
                eng_stress, eng_strain, uts_dic = convert_fvd_engss(df=fvd, geometry=self.geom_dic, paths=self.path_dic)
                # ##ADD RAW DATA PATH TO UTS DIC
                material_name = Path(self.path_dic['exp_fvd']).stem
                uts_dic['RAW_DATA'] = self.path_dic['exp_fvd']
                uts_dic['material'] = material_name
                self.textArea.appendPlainText(f"""\n Analysis First Phase:\nEngineering Stress: \n{str(eng_stress)}\nEngineering Strain: \n{str(eng_strain)} \nUTS Dictionary: \n{str(uts_dic)} \n""")  # Displaying part of the results
                
                # ##CALCULATE TRUE STRESS AND TRUE STRAIN FROM ENG STRESS-STRAIN
                true_strain, true_stress = true_stress_strain(eng_stress=eng_stress, eng_strain=eng_strain)
                # ##CREATE TRUE DF
                true_df = pd.concat([true_strain, true_stress], axis=1, keys=['TRUE_STRAIN', 'TRUE_STRESS'])
                all_df = pd.concat([fvd, true_df], axis=1)
                df_ss_file = os.path.join(self.path_dic['curr_results'], 'DF_SS.csv')
                all_df.to_csv(df_ss_file, index=False)
                # ##PLOT TRUE STRESS-STRAIN
                true_stress_strain_plot = true_stress_true_strain(x=true_strain, y=true_stress, **self.path_dic)
                # true_stress_strain_plot = os.path.join(self.path_dic['curr_results'], 'TRUE_SS.png')
                # self.displayPlot(true_stress_strain_plot)
                self.plotPaths.append(true_stress_strain_plot)
                # Update UTS dictionary with the final true stress and strain
                uts_dic = merge_dicts(uts_dic, {'TRUE_STRAIN': true_strain.iloc[-1], 'TRUE_STRESS': true_stress.iloc[-1]})

                # Calculate the slope of the true stress-strain curve
                min_slope, max_slope = aoc_calc_slope(true_strain, true_stress)

                # Calculate second derivative
                strain_idx = np.where(true_strain <= 0.002)

                # Interpolate between each point of the curve
                func = interp1d(true_strain.values, true_stress.values, kind='linear')
                interp_strain = np.arange(true_strain.iloc[0], true_strain.iloc[-1], 1e-4)
                interp_stress = func(interp_strain)

                true_vs_interp_plot = os.path.join(self.path_dic['curr_results'], 'True_vs_Interpolated.png')
                # Compare engineering with interpolated
                true_vs_interpolated_path = compare_interp_true(truex=true_strain,
                                    truey=true_stress,
                                    interpx=interp_strain,
                                    interpy=interp_stress,
                                    kind=func._kind,
                                    **self.path_dic)
                # self.displayPlot(plot_path)
                self.plotPaths.append(true_vs_interpolated_path)
                # Export interpolated data up to UTS position as DataFrame
                tdf = pd.DataFrame(columns=['TRUE_STRAIN', 'TRUE_STRESS'],
                                data=np.stack((interp_strain, interp_stress), axis=1))
                true_interp_file = os.path.join(self.path_dic['curr_results'], 'TRUE_INTERP.csv')
                uts_dic['TRUE_TO_UTS'] = true_interp_file
                tdf.to_csv(uts_dic['TRUE_TO_UTS'], index=False)

                # Iterate range of window sizes and find the best R² score
                sav_dic = {}
                for j, wl in enumerate([x for x in np.arange(11, 101, 3) if x % 2 != 0]):
                    sder_strain = savgol_filter(interp_strain, window_length=wl, polyorder=3, deriv=2)
                    zero = np.abs(sder_strain - 0.0).argmin()
                    mod_strain = interp_strain[:zero].reshape(-1, 1)
                    mod_stress = interp_stress[:zero].reshape(-1, 1)

                    if len(mod_strain) > 5:
                        model = LinearRegression().fit(mod_strain, mod_stress)
                        prediction = model.predict(mod_strain)
                        linear = pd.DataFrame(data=np.stack((mod_strain.flatten() * 100, mod_stress.flatten(), prediction.flatten()), axis=1),
                                            columns=['TRUE_STRAIN', 'TRUE_STRESS', 'PRED_TRUE_STRESS'])
                        linear.to_csv(os.path.join(self.path_dic['curr_results'], 'LINEAR_REGION_WL%s.csv' % (wl)), index=False)

                        mape = max_error(mod_stress, prediction)
                        sav_dic[wl] = {'E': model.coef_[0][0],
                                    'SIGMA_Y': interp_stress[zero],
                                    'r2': round(r2_score(mod_stress, prediction), 3),
                                    'MAPE': mape,
                                    'SEC_DER': sder_strain,
                                    'ZERO': zero,
                                    'WINDOW_LENGTH': wl}
                # Export `sav_dic` to JSON for review and plot each WL analysis
                for k in sav_dic.keys():
                    plot_sec_der_peaks_path_1 = plot_sec_der_peaks(true_strain=true_strain,
                                       true_stress=true_stress,
                                       interp_strain=interp_strain,
                                       interp_stress=interp_stress,
                                       img_name='SEC_DER_%s' % (k),
                                       data_dic=sav_dic[k],
                                       **self.path_dic)
                    self.plotPaths.append(plot_sec_der_peaks_path_1)

                # Create DataFrame from dictionary and transpose
                df_dic = pd.DataFrame(copy.deepcopy(sav_dic)).transpose()

                # Filter results to include only R² >= 0.95
                df_dic = df_dic[df_dic['r2'] >= 0.95]
                df_dic['MULTI'] = df_dic['r2'] / (df_dic['SIGMA_Y'] * df_dic['MAPE'])

                # Sort DataFrame by yield strength
                df_dic.sort_values(['MULTI'], ascending=[True], inplace=True)

                # Export DataFrame for manual checks
                df_dic.to_csv(os.path.join(self.path_dic['curr_results'], 'WINDOW_LENGTH.csv'),
                              columns=[c for c in df_dic.columns if 'SEC_DER' not in c])

                # Get the top 3 and select the maximum yield value as 'best' value
                best_key = df_dic.iloc[0].name
                best = sav_dic[best_key]

                # Add Young's modulus and yield strength to UTS dictionary
                uts_dic = merge_dicts(uts_dic, {'E': best['E'], 'SIGMA_Y': best['SIGMA_Y']})

                # Plot the second derivative outputs (including modulus line)
                plot_sec_der_peaks_path_2 = plot_sec_der_peaks(true_strain=true_strain,
                                   true_stress=true_stress,
                                   interp_strain=interp_strain,
                                   interp_stress=interp_stress,
                                   img_name='SEC_DER',
                                   data_dic=best,
                                   **self.path_dic)
                self.plotPaths.append(plot_sec_der_peaks_path_2)

                # Plot the linear region
                linear_plot_path = plot_linear(true_strain=true_strain,
                            true_stress=true_stress,
                            interp_strain=interp_strain,
                            interp_stress=interp_stress,
                            img_name='LINEAR',
                            data_dic=best,
                            **self.path_dic)
                # self.displayPlot(os.path.join(self.path_dic['curr_results'], 'LINEAR.png'))  # Display this plot
                self.plotPaths.append(linear_plot_path)

                ###################################
                # Plastic Strain up to UTS
                ###################################
                odf = pd.DataFrame(data={'TRUE_STRAIN': interp_strain, 'TRUE_STRESS': interp_stress})
                op = odf[odf['TRUE_STRESS'] >= best['SIGMA_Y']].copy()

                # Modify strain to be zero at yield stress
                op['PLASTIC_STRAIN'] = op['TRUE_STRAIN'] - (op['TRUE_STRESS'] / best['E'])
                op['PLASTIC_STRAIN'] = op['PLASTIC_STRAIN'] - op['PLASTIC_STRAIN'].iloc[0]

                # Replace any negative strains with very low strain
                op['PLASTIC_STRAIN'] = np.where(op['PLASTIC_STRAIN'] < 0, 1E-20, op['PLASTIC_STRAIN'])

                # Write plastic strain to CSV
                op.to_csv(os.path.join(self.path_dic['curr_results'], 'ABA_TSPE_UTS.csv'), index=False)

                ###################################
                # Extend data beyond UTS
                ###################################
                m_range = np.arange(0, max_slope + 50, 50)
                slope_dic = {}

                for j, m in enumerate(m_range):
                    # ##THE INTERCEPT IS A FUNCTION OF SIGMA TRUE UTS AND
                    # ##EPSILON TRUE UTS (C = SIG_T - M*ESP_T)
                    c = uts_dic['TRUE_STRESS'] - (m * uts_dic['TRUE_STRAIN'])
                    slope_dic[m] = {'Y_INTERCEPT': c, 'ABAQUS_PLASTIC': os.path.join(self.path_dic['curr_results'], f'ABA_M{int(m)}.csv')}
                    # ##SET STRAIN RANGE TO BE AT LEAST 1000 ELEMENTS IN SIZE
                    estrain = np.linspace(interp_strain[-1] + 1e-4, 2, num=1000).reshape(-1, 1)
                    estress = m * estrain + c
                    # ##COMBINE ORIGINAL AND EXTENDED DATA TO GET THE FULL MATERIAL PROPERTIES FOR ABAQUS
                    df = pd.DataFrame(data={'TRUE_STRAIN': np.concatenate((interp_strain, estrain.flatten()), axis=0),
                                            'TRUE_STRESS': np.concatenate((interp_stress, estress.flatten()), axis=0)})
                    # ##ABAQUS REQUIRES STRAIN AND STRESS TO START FROM YIELD POSITION
                    # ## WE NEED TO MODIFY THE TRUE STRESS - TRUE STRAIN TO TRUE STRESS - PLASTIC STRAIN
                    plastic = df[df['TRUE_STRESS'] >= best['SIGMA_Y']].copy()
                    # ##MODIFY STRAIN TO BE ZERO AT YIELD STRESS
                    plastic['PLASTIC_STRAIN'] = plastic['TRUE_STRAIN'] - (plastic['TRUE_STRESS'] / best['E'])
                    # ##RESET STRAIN TO BE ZERO AT YIELD
                    plastic['PLASTIC_STRAIN'] = plastic['PLASTIC_STRAIN'] - plastic['PLASTIC_STRAIN'].iloc[0]
                    # ##REPLACE ANY NEGATIVE STRAINS WITH VERY LOW STRAIN
                    plastic['PLASTIC_STRAIN'] = np.where(plastic['PLASTIC_STRAIN'] < 0, 1E-20, plastic['PLASTIC_STRAIN'])
                    # ##DROP TRUE STRAIN
                    plastic.drop('TRUE_STRAIN', axis=1, inplace=True)
                    # ##SAVE ABAQUS DATA TO CSV FILE
                    plastic.to_csv(slope_dic[m]['ABAQUS_PLASTIC'], index=False)
                    # ##PLOT TRUE STRESS PLASTIC STRAIN
                    true_stress_plastic_strain(x=plastic['PLASTIC_STRAIN'], y=plastic['TRUE_STRESS'], name=f'TS_EP_M{int(m)}', **self.path_dic)

                uts_dic = merge_dicts(uts_dic, {'SLOPE': slope_dic})
                material_properties_file = os.path.join(self.path_dic['curr_results'], f"{uts_dic['material']}_properties.txt")
                write_json_file(dic=uts_dic, pth=self.path_dic['curr_results'], filename=material_properties_file)
                path_dic_file = os.path.join(self.path_dic['curr_results'], 'PATH_DIC.txt')
                write_json_file(dic=self.path_dic, pth=self.path_dic['curr_results'], filename=path_dic_file)
                all_slopes_plot = os.path.join(self.path_dic['curr_results'], 'All_Slopes.png')
                plot_all_slopes_path = plot_all_slopes(true_strain=true_strain, true_stress=true_stress, m_range=m_range, uts_dic=uts_dic, path_dic=self.path_dic)
                # self.displayPlot(os.path.join(self.path_dic['curr_results'], 'All_Slopes.png'))  # Display this plot
                self.plotPaths.append(plot_all_slopes_path)

                # List all generated files with full paths
                file_summary = f"\nGenerated Files:\n"
                file_summary += f"1. DF_SS.csv: {df_ss_file}\n"
                file_summary += f"2. TRUE_INTERP.csv: {true_interp_file}\n"
                file_summary += f"3. LINEAR_REGION files: see OUTPUT folder\n"
                file_summary += f"4. WINDOW_LENGTH.csv: {os.path.join(self.path_dic['curr_results'], 'WINDOW_LENGTH.csv')}\n"
                file_summary += f"5. ABA_TSPE_UTS.csv: {os.path.join(self.path_dic['curr_results'], 'ABA_TSPE_UTS.csv')}\n"
                file_summary += f"6. ABA_M<value>.csv files: see OUTPUT folder\n"
                file_summary += f"7. Material Properties (JSON): {material_properties_file}\n"
                file_summary += f"8. PATH_DIC.txt: {path_dic_file}\n"

                # List all generated plots with full paths
                plots_summary = f"\nGenerated Plots:\n"
                plots_summary += f"1. True Stress-Strain Curve: {true_stress_strain_plot}\n"
                plots_summary += f"2. True vs. Interpolated Stress-Strain: {true_vs_interp_plot}\n"
                plots_summary += f"3. Second Derivative Peaks (Best Window): {os.path.join(self.path_dic['curr_results'], 'SEC_DER.png')}\n"
                plots_summary += f"4. Linear Region (Best Window): {os.path.join(self.path_dic['curr_results'], 'LINEAR.png')}\n"
                plots_summary += f"5. True Stress-Plastic Strain Curves (TS_EP_M<value>.png): see OUTPUT folder\n"
                plots_summary += f"6. All Slopes on True Stress-Strain Curve: {all_slopes_plot}\n"

                self.textArea.appendPlainText(file_summary)
                self.textArea.appendPlainText(plots_summary)
                # self.displayPlot(true_stress_strain_plot)
                self.stopLoadingAnimation()
                self.statusLabel.setText("Analysis completed successfully.")
                if self.plotPaths:
                    self.prevButton.setEnabled(False)
                    self.nextButton.setEnabled(len(self.plotPaths) > 1)
                    self.saveOutputsButton.setEnabled(True)
                    self.saveAllButton.setEnabled(True)
                    self.displayPlot()
            except Exception as e:
                self.statusLabel.setText(f"Failed to analyze data: {str(e)}")
                self.prevButton.setEnabled(False)
                self.nextButton.setEnabled(False)
                self.saveOutputsButton.setEnabled(False)
                self.saveAllButton.setEnabled(False)
        else:
            self.statusLabel.setText("No file selected for analysis.")

        
    def clearContents(self):
        self.textArea.clear()
        self.statusLabel.setText("Status: Waiting for data...")
        self.figure.clear()
        self.canvas.draw()
        self.analyzeButton.setEnabled(False)
        self.prevButton.setEnabled(False)
        self.nextButton.setEnabled(False)
        self.saveOutputsButton.setEnabled(False)
        self.saveAllButton.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    app_window = AppWindow()
    app_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()