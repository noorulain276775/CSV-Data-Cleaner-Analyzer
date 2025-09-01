#!/usr/bin/env python3
"""
CSV Data Cleaner Analyzer
Main entry point for the application
"""

import sys
from PyQt5.QtWidgets import QApplication
from main_screen import MainWindow

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("CSV Data Cleaner Analyzer")
    app.setApplicationVersion("1.0.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

