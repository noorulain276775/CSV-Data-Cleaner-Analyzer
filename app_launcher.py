#!/usr/bin/env python3
"""
CSV Data Cleaner Analyzer - Application Launcher
Main entry point with version selection
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from main_screen import MainWindow
from enhanced_gui import EnhancedMainWindow

class VersionSelector(QMainWindow):
    """Simple window to select which version to run"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Data Cleaner Analyzer - Version Selector")
        self.setGeometry(300, 300, 400, 200)
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("CSV Data Cleaner Analyzer")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Choose your version:")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Buttons
        basic_btn = QPushButton("Basic Version")
        basic_btn.setFont(QFont("Arial", 11))
        basic_btn.clicked.connect(self.run_basic)
        layout.addWidget(basic_btn)
        
        enhanced_btn = QPushButton("Enhanced Version")
        enhanced_btn.setFont(QFont("Arial", 11))
        enhanced_btn.clicked.connect(self.run_enhanced)
        layout.addWidget(enhanced_btn)
        
        # Info
        info_label = QLabel("Enhanced version includes advanced features like:\n• Data quality reporting\n• Outlier detection\n• Advanced cleaning options\n• Data export in multiple formats")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)
        
    def run_basic(self):
        """Run the basic version"""
        self.hide()
        self.basic_window = MainWindow()
        self.basic_window.show()
        
    def run_enhanced(self):
        """Run the enhanced version"""
        self.hide()
        self.enhanced_window = EnhancedMainWindow()
        self.enhanced_window.show()

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("CSV Data Cleaner Analyzer")
    app.setApplicationVersion("1.0.0")
    
    # Check if command line arguments specify version
    if len(sys.argv) > 1:
        version = sys.argv[1].lower()
        if version in ['basic', '--basic', '-b']:
            window = MainWindow()
        elif version in ['enhanced', '--enhanced', '-e']:
            window = EnhancedMainWindow()
        else:
            print("Usage: python app_launcher.py [basic|enhanced]")
            print("Default: Shows version selector")
            window = VersionSelector()
    else:
        # Show version selector by default
        window = VersionSelector()
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
