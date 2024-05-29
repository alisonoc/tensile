import sys
import pandas as pd

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from MATERIAL_CHECK import material_check

class MaterialAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Material Analysis Tool")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()

        self.uploadButton = QPushButton("Upload Data")
        self.uploadButton.clicked.connect(self.uploadData)

        self.analyzeButton = QPushButton("Analyze")
        self.analyzeButton.clicked.connect(self.analyzeData)

        self.statusLabel = QLabel("Status: Waiting for data...")

        self.plotCanvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.ax = self.plotCanvas.figure.subplots()

        self.layout.addWidget(self.uploadButton)
        self.layout.addWidget(self.analyzeButton)
        self.layout.addWidget(self.statusLabel)
        self.layout.addWidget(self.plotCanvas)

        self.centralWidget.setLayout(self.layout)

    def uploadData(self):
        # Implement file upload logic
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if fileName:
            print(f"Selected file: {fileName}")
            # Load the CSV file and display its contents
            df = pd.read_csv(fileName)
            # self.textArea.setText(df.to_string())

    def analyzeData(self):
        # Integrate your analysis code here
        pass

    def plotResults(self):
        # Use this function to update the plot with analysis results
        pass

def main():
    app = QApplication(sys.argv)
    mainWindow = MaterialAnalysisApp()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
