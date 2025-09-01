#!/usr/bin/env python3
"""
Enhanced CSV Data Cleaner Analyzer - GUI Implementation
Main screen with advanced features and utility functions
"""

import os
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QComboBox, QLineEdit, QTextEdit,
                             QGroupBox, QGridLayout, QMessageBox, QTabWidget,
                             QProgressBar, QSpinBox, QCheckBox, QSplitter,
                             QListWidget, QListWidgetItem, QDialog, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from utils import *

class ExportDialog(QDialog):
    """Dialog for exporting data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Data")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QFormLayout(self)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['CSV', 'Excel', 'JSON', 'Parquet'])
        
        self.file_path_label = QLabel("No file selected")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.accept)
        
        layout.addRow("Format:", self.format_combo)
        layout.addRow("File:", self.file_path_label)
        layout.addRow("", self.browse_btn)
        layout.addRow("", self.export_btn)
        
    def browse_file(self):
        format_map = {
            'CSV': 'csv',
            'Excel': 'xlsx',
            'JSON': 'json',
            'Parquet': 'parquet'
        }
        
        format_ext = format_map[self.format_combo.currentText()]
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", f"{self.format_combo.currentText()} Files (*.{format_ext})"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            
    def get_export_info(self):
        return {
            'format': self.format_combo.currentText().lower(),
            'file_path': self.file_path_label.text()
        }

class AdvancedCleaningDialog(QDialog):
    """Dialog for advanced data cleaning options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Data Cleaning")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Duplicate removal
        duplicate_group = QGroupBox("Duplicate Removal")
        duplicate_layout = QFormLayout()
        
        self.duplicate_strategy = QComboBox()
        self.duplicate_strategy.addItems(['Remove all duplicates', 'Keep first', 'Keep last'])
        
        duplicate_layout.addRow("Strategy:", self.duplicate_strategy)
        duplicate_group.setLayout(duplicate_layout)
        layout.addWidget(duplicate_group)
        
        # Missing value handling
        missing_group = QGroupBox("Missing Value Handling")
        missing_layout = QFormLayout()
        
        self.missing_strategy = QComboBox()
        self.missing_strategy.addItems(['Mean', 'Median', 'Mode', 'Forward fill', 'Backward fill'])
        
        missing_layout.addRow("Strategy:", self.missing_strategy)
        missing_group.setLayout(missing_layout)
        layout.addWidget(missing_group)
        
        # Column cleaning
        column_group = QGroupBox("Column Cleaning")
        column_layout = QFormLayout()
        
        self.clean_names_check = QCheckBox("Clean column names")
        self.clean_names_check.setChecked(True)
        
        self.detect_types_check = QCheckBox("Detect optimal data types")
        self.detect_types_check.setChecked(True)
        
        column_layout.addRow(self.clean_names_check)
        column_layout.addRow(self.detect_types_check)
        column_group.setLayout(column_layout)
        layout.addWidget(column_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
    def get_cleaning_options(self):
        return {
            'duplicate_strategy': self.duplicate_strategy.currentText(),
            'missing_strategy': self.missing_strategy.currentText(),
            'clean_names': self.clean_names_check.isChecked(),
            'detect_types': self.detect_types_check.isChecked()
        }

class EnhancedMainWindow(QMainWindow):
    """Enhanced main window for the CSV Data Cleaner Analyzer"""
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.original_df = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Enhanced CSV Data Cleaner Analyzer")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create title
        title_label = QLabel("Enhanced CSV Data Cleaner Analyzer")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create file loading section
        self.create_file_section(main_layout)
        
        # Create tab widget for different operations
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_data_view_tab()
        self.create_cleaning_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_advanced_tab()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
    def create_file_section(self, layout):
        """Create the file loading section"""
        file_group = QGroupBox("Load CSV File")
        file_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        
        load_button = QPushButton("Browse")
        load_button.clicked.connect(self.load_csv)
        
        export_button = QPushButton("Export")
        export_button.clicked.connect(self.export_data)
        export_button.setEnabled(False)
        self.export_button = export_button
        
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(load_button)
        file_layout.addWidget(export_button)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
    def create_data_view_tab(self):
        """Create the data viewing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Data info section
        info_group = QGroupBox("Data Information")
        info_layout = QGridLayout()
        
        self.row_count_label = QLabel("Rows: 0")
        self.col_count_label = QLabel("Columns: 0")
        self.memory_label = QLabel("Memory: 0 MB")
        self.duplicate_label = QLabel("Duplicates: 0")
        
        info_layout.addWidget(self.row_count_label, 0, 0)
        info_layout.addWidget(self.col_count_label, 0, 1)
        info_layout.addWidget(self.memory_label, 0, 2)
        info_layout.addWidget(self.duplicate_label, 0, 3)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Data table
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        self.tab_widget.addTab(tab, "Data View")
        
    def create_cleaning_tab(self):
        """Create the data cleaning tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Basic cleaning section
        basic_group = QGroupBox("Basic Cleaning")
        basic_layout = QGridLayout()
        
        self.nulls_combo = QComboBox()
        self.nulls_combo.addItems(["any", "all"])
        self.nulls_combo.setCurrentText("any")
        
        remove_nulls_btn = QPushButton("Remove Nulls")
        remove_nulls_btn.clicked.connect(self.remove_nulls)
        
        basic_layout.addWidget(QLabel("Strategy:"), 0, 0)
        basic_layout.addWidget(self.nulls_combo, 0, 1)
        basic_layout.addWidget(remove_nulls_btn, 0, 2)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Advanced cleaning section
        advanced_cleaning_btn = QPushButton("Advanced Cleaning Options")
        advanced_cleaning_btn.clicked.connect(self.show_advanced_cleaning)
        layout.addWidget(advanced_cleaning_btn)
        
        # Filter section
        filter_group = QGroupBox("Filter Data")
        filter_layout = QGridLayout()
        
        self.filter_column_combo = QComboBox()
        self.filter_operator_combo = QComboBox()
        self.filter_operator_combo.addItems(["==", "!=", ">", "<", ">=", "<=", "contains"])
        self.filter_value_input = QLineEdit()
        
        filter_btn = QPushButton("Apply Filter")
        filter_btn.clicked.connect(self.apply_filter)
        
        filter_layout.addWidget(QLabel("Column:"), 0, 0)
        filter_layout.addWidget(self.filter_column_combo, 0, 1)
        filter_layout.addWidget(QLabel("Operator:"), 0, 2)
        filter_layout.addWidget(self.filter_operator_combo, 0, 3)
        filter_layout.addWidget(QLabel("Value:"), 0, 4)
        filter_layout.addWidget(self.filter_value_input, 0, 5)
        filter_layout.addWidget(filter_btn, 0, 6)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Reset button
        reset_btn = QPushButton("Reset to Original Data")
        reset_btn.clicked.connect(self.reset_data)
        layout.addWidget(reset_btn)
        
        self.tab_widget.addTab(tab, "Data Cleaning")
        
    def create_analysis_tab(self):
        """Create the data analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Summary statistics
        summary_group = QGroupBox("Summary Statistics")
        summary_layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        
        summary_btn = QPushButton("Generate Summary")
        summary_btn.clicked.connect(self.generate_summary)
        
        summary_layout.addWidget(summary_btn)
        summary_layout.addWidget(self.summary_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Column analysis
        column_group = QGroupBox("Column Analysis")
        column_layout = QGridLayout()
        
        self.column_combo = QComboBox()
        analyze_column_btn = QPushButton("Analyze Column")
        analyze_column_btn.clicked.connect(self.analyze_column)
        
        column_layout.addWidget(QLabel("Select Column:"), 0, 0)
        column_layout.addWidget(self.column_combo, 0, 1)
        column_layout.addWidget(analyze_column_btn, 0, 2)
        
        self.column_analysis_text = QTextEdit()
        self.column_analysis_text.setReadOnly(True)
        column_layout.addWidget(self.column_analysis_text, 1, 0, 1, 3)
        
        column_group.setLayout(column_layout)
        layout.addWidget(column_group)
        
        self.tab_widget.addTab(tab, "Data Analysis")
        
    def create_visualization_tab(self):
        """Create the data visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Chart controls
        controls_group = QGroupBox("Chart Controls")
        controls_layout = QGridLayout()
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart"])
        
        self.x_column_combo = QComboBox()
        self.y_column_combo = QComboBox()
        
        create_chart_btn = QPushButton("Create Chart")
        create_chart_btn.clicked.connect(self.create_chart)
        
        controls_layout.addWidget(QLabel("Chart Type:"), 0, 0)
        controls_layout.addWidget(self.chart_type_combo, 0, 1)
        controls_layout.addWidget(QLabel("X Column:"), 0, 2)
        controls_layout.addWidget(self.x_column_combo, 0, 3)
        controls_layout.addWidget(QLabel("Y Column:"), 0, 4)
        controls_layout.addWidget(self.y_column_combo, 0, 5)
        controls_layout.addWidget(create_chart_btn, 0, 6)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Chart display
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.tab_widget.addTab(tab, "Visualization")
        
    def create_advanced_tab(self):
        """Create the advanced features tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Data quality report
        quality_group = QGroupBox("Data Quality Report")
        quality_layout = QVBoxLayout()
        
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        
        quality_btn = QPushButton("Generate Quality Report")
        quality_btn.clicked.connect(self.generate_quality_report)
        
        quality_layout.addWidget(quality_btn)
        quality_layout.addWidget(self.quality_text)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        # Outlier detection
        outlier_group = QGroupBox("Outlier Detection")
        outlier_layout = QGridLayout()
        
        self.outlier_column_combo = QComboBox()
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["IQR", "Z-Score"])
        
        detect_outliers_btn = QPushButton("Detect Outliers")
        detect_outliers_btn.clicked.connect(self.detect_outliers)
        
        outlier_layout.addWidget(QLabel("Column:"), 0, 0)
        outlier_layout.addWidget(self.outlier_column_combo, 0, 1)
        outlier_layout.addWidget(QLabel("Method:"), 0, 2)
        outlier_layout.addWidget(self.outlier_method_combo, 0, 3)
        outlier_layout.addWidget(detect_outliers_btn, 0, 4)
        
        self.outlier_text = QTextEdit()
        self.outlier_text.setReadOnly(True)
        outlier_layout.addWidget(self.outlier_text, 1, 0, 1, 5)
        
        outlier_group.setLayout(outlier_layout)
        layout.addWidget(outlier_group)
        
        self.tab_widget.addTab(tab, "Advanced Features")
        
    def load_csv(self):
        """Load a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.original_df = self.df.copy()
                self.file_path_label.setText(file_path)
                
                self.update_data_display()
                self.update_column_combos()
                self.export_button.setEnabled(True)
                self.statusBar().showMessage(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")
                
    def update_data_display(self):
        """Update the data table display"""
        if self.df is None:
            return
            
        # Update info labels
        self.row_count_label.setText(f"Rows: {len(self.df):,}")
        self.col_count_label.setText(f"Columns: {len(self.df.columns)}")
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        self.memory_label.setText(f"Memory: {memory_mb:.2f} MB")
        
        duplicate_count = self.df.duplicated().sum()
        self.duplicate_label.setText(f"Duplicates: {duplicate_count}")
        
        # Update data table
        self.data_table.setRowCount(min(1000, len(self.df)))
        self.data_table.setColumnCount(len(self.df.columns))
        self.data_table.setHorizontalHeaderLabels(self.df.columns)
        
        # Populate table with data (limit to first 1000 rows for performance)
        display_df = self.df.head(1000)
        for i in range(len(display_df)):
            for j in range(len(display_df.columns)):
                value = display_df.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
                
        if len(self.df) > 1000:
            self.statusBar().showMessage(f"Showing first 1000 rows of {len(self.df):,} total rows")
            
    def update_column_combos(self):
        """Update all column combo boxes"""
        if self.df is None:
            return
            
        columns = list(self.df.columns)
        
        # Update filter column combo
        self.filter_column_combo.clear()
        self.filter_column_combo.addItems(columns)
        
        # Update analysis column combo
        self.column_combo.clear()
        self.column_combo.addItems(columns)
        
        # Update visualization column combos
        self.x_column_combo.clear()
        self.x_column_combo.addItems(columns)
        self.y_column_combo.clear()
        self.y_column_combo.addItems(columns)
        
        # Update outlier column combo
        self.outlier_column_combo.clear()
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.outlier_column_combo.addItems(numeric_columns)
        
    def remove_nulls(self):
        """Remove null values from the dataset"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        strategy = self.nulls_combo.currentText()
        how = strategy
        
        try:
            old_count = len(self.df)
            self.df = self.df.dropna(how=how)
            new_count = len(self.df)
            
            self.update_data_display()
            self.statusBar().showMessage(f"Removed {old_count - new_count} rows with null values")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove nulls: {str(e)}")
            
    def show_advanced_cleaning(self):
        """Show advanced cleaning options dialog"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        dialog = AdvancedCleaningDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            options = dialog.get_cleaning_options()
            self.apply_advanced_cleaning(options)
            
    def apply_advanced_cleaning(self, options):
        """Apply advanced cleaning options"""
        try:
            # Duplicate removal
            if options['duplicate_strategy'] == 'Remove all duplicates':
                self.df = remove_duplicates(self.df, keep=False)
            elif options['duplicate_strategy'] == 'Keep first':
                self.df = remove_duplicates(self.df, keep='first')
            elif options['duplicate_strategy'] == 'Keep last':
                self.df = remove_duplicates(self.df, keep='last')
                
            # Missing value handling
            strategy_map = {
                'Mean': 'mean',
                'Median': 'median',
                'Mode': 'mode',
                'Forward fill': 'forward',
                'Backward fill': 'backward'
            }
            
            if options['missing_strategy'] in strategy_map:
                self.df = fill_missing_values(self.df, strategy=strategy_map[options['missing_strategy']])
                
            # Column name cleaning
            if options['clean_names']:
                self.df = clean_column_names(self.df)
                
            # Data type detection
            if options['detect_types']:
                type_suggestions = detect_data_types(self.df)
                # Note: In a real application, you might want to show these suggestions to the user
                
            self.update_data_display()
            self.update_column_combos()
            self.statusBar().showMessage("Advanced cleaning applied successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply advanced cleaning: {str(e)}")
            
    def apply_filter(self):
        """Apply filter to the dataset"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        column = self.filter_column_combo.currentText()
        operator = self.filter_operator_combo.currentText()
        value = self.filter_value_input.text()
        
        if not column or not value:
            QMessageBox.warning(self, "Warning", "Please select a column and enter a value")
            return
            
        try:
            # Convert value to appropriate type
            if self.df[column].dtype in ['int64', 'float64']:
                try:
                    value = float(value)
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Please enter a valid number for numeric columns")
                    return
                    
            old_count = len(self.df)
            
            # Apply filter
            if operator == '==':
                self.df = self.df[self.df[column] == value]
            elif operator == '!=':
                self.df = self.df[self.df[column] != value]
            elif operator == '>':
                self.df = self.df[self.df[column] > value]
            elif operator == '<':
                self.df = self.df[self.df[column] < value]
            elif operator == '>=':
                self.df = self.df[self.df[column] >= value]
            elif operator == '<=':
                self.df = self.df[self.df[column] <= value]
            elif operator == 'contains':
                self.df = self.df[self.df[column].astype(str).str.contains(str(value), na=False)]
                
            new_count = len(self.df)
            self.update_data_display()
            self.statusBar().showMessage(f"Filter applied: {old_count - new_count} rows removed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filter: {str(e)}")
            
    def reset_data(self):
        """Reset data to original loaded state"""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.update_data_display()
            self.update_column_combos()
            self.statusBar().showMessage("Data reset to original state")
        else:
            QMessageBox.warning(self, "Warning", "No original data to reset to")
            
    def generate_summary(self):
        """Generate summary statistics"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        try:
            summary = self.df.describe()
            summary_str = summary.to_string()
            
            # Add additional info
            summary_str += f"\n\nDataset Shape: {self.df.shape}"
            summary_str += f"\nData Types:\n{self.df.dtypes.to_string()}"
            summary_str += f"\n\nMissing Values:\n{self.df.isnull().sum().to_string()}"
            
            self.summary_text.setText(summary_str)
            self.statusBar().showMessage("Summary generated successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate summary: {str(e)}")
            
    def generate_quality_report(self):
        """Generate data quality report"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        try:
            report = generate_data_report(self.df)
            
            report_str = "DATA QUALITY REPORT\n"
            report_str += "=" * 50 + "\n\n"
            
            report_str += f"Dataset Shape: {report['shape']}\n"
            report_str += f"Memory Usage: {report['memory_usage']:.2f} MB\n"
            report_str += f"Duplicate Rows: {report['duplicate_rows']} ({report['duplicate_percentage']:.2f}%)\n\n"
            
            report_str += "MISSING VALUES:\n"
            report_str += "-" * 20 + "\n"
            for col, count in report['missing_values'].items():
                if count > 0:
                    percentage = report['missing_percentage'][col]
                    report_str += f"{col}: {count} ({percentage:.2f}%)\n"
                    
            report_str += "\nCOLUMN TYPES:\n"
            report_str += "-" * 20 + "\n"
            for col, dtype in report['data_types'].items():
                report_str += f"{col}: {dtype}\n"
                
            if report['numeric_columns']:
                report_str += f"\nNUMERIC COLUMNS: {', '.join(report['numeric_columns'])}\n"
                
            if report['categorical_columns']:
                report_str += f"\nCATEGORICAL COLUMNS: {', '.join(report['categorical_columns'])}\n"
                
            self.quality_text.setText(report_str)
            self.statusBar().showMessage("Quality report generated successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate quality report: {str(e)}")
            
    def detect_outliers(self):
        """Detect outliers in selected column"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        column = self.outlier_column_combo.currentText()
        method = self.outlier_method_combo.currentText().lower()
        
        if not column:
            return
            
        try:
            outliers = detect_outliers(self.df, columns=[column], method=method)
            
            if column in outliers and outliers[column]:
                outlier_count = len(outliers[column])
                outlier_indices = outliers[column]
                
                report_str = f"OUTLIER DETECTION REPORT\n"
                report_str += "=" * 40 + "\n\n"
                report_str += f"Column: {column}\n"
                report_str += f"Method: {method.upper()}\n"
                report_str += f"Outliers Found: {outlier_count}\n\n"
                
                report_str += "Outlier Values:\n"
                for idx in outlier_indices[:20]:  # Show first 20
                    report_str += f"Row {idx}: {self.df.loc[idx, column]}\n"
                    
                if outlier_count > 20:
                    report_str += f"\n... and {outlier_count - 20} more outliers\n"
                    
                self.outlier_text.setText(report_str)
                self.statusBar().showMessage(f"Found {outlier_count} outliers in {column}")
            else:
                self.outlier_text.setText(f"No outliers found in {column} using {method.upper()} method.")
                self.statusBar().showMessage(f"No outliers found in {column}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to detect outliers: {str(e)}")
            
    def analyze_column(self):
        """Analyze a specific column"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        column = self.column_combo.currentText()
        if not column:
            return
            
        try:
            col_data = self.df[column]
            analysis = []
            
            analysis.append(f"Column: {column}")
            analysis.append(f"Data Type: {col_data.dtype}")
            analysis.append(f"Total Values: {len(col_data)}")
            analysis.append(f"Non-Null Values: {col_data.count()}")
            analysis.append(f"Null Values: {col_data.isnull().sum()}")
            analysis.append(f"Unique Values: {col_data.nunique()}")
            
            if col_data.dtype in ['int64', 'float64']:
                analysis.append(f"Mean: {col_data.mean():.4f}")
                analysis.append(f"Median: {col_data.median():.4f}")
                analysis.append(f"Std: {col_data.std():.4f}")
                analysis.append(f"Min: {col_data.min()}")
                analysis.append(f"Max: {col_data.max()}")
                
            if col_data.dtype == 'object':
                analysis.append(f"Most Common Values:\n{col_data.value_counts().head(10).to_string()}")
                
            self.column_analysis_text.setText("\n".join(analysis))
            self.statusBar().showMessage(f"Column analysis completed for {column}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze column: {str(e)}")
            
    def create_chart(self):
        """Create a chart based on selected options"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        chart_type = self.chart_type_combo.currentText()
        x_column = self.x_column_combo.currentText()
        y_column = self.y_column_combo.currentText()
        
        if not x_column:
            QMessageBox.warning(self, "Warning", "Please select an X column")
            return
            
        try:
            self.ax.clear()
            
            if chart_type == "Histogram":
                self.df[x_column].hist(ax=self.ax, bins=30)
                self.ax.set_title(f"Histogram of {x_column}")
                self.ax.set_xlabel(x_column)
                self.ax.set_ylabel("Frequency")
                
            elif chart_type == "Bar Chart":
                if self.df[x_column].dtype == 'object':
                    value_counts = self.df[x_column].value_counts().head(20)
                    value_counts.plot(kind='bar', ax=self.ax)
                    self.ax.set_title(f"Bar Chart of {x_column}")
                    self.ax.set_xlabel(x_column)
                    self.ax.set_ylabel("Count")
                    plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
                else:
                    QMessageBox.warning(self, "Warning", "Bar charts work best with categorical data")
                    return
                    
            elif chart_type == "Scatter Plot":
                if not y_column:
                    QMessageBox.warning(self, "Warning", "Scatter plot requires both X and Y columns")
                    return
                if self.df[x_column].dtype in ['int64', 'float64'] and self.df[y_column].dtype in ['int64', 'float64']:
                    self.ax.scatter(self.df[x_column], self.df[y_column], alpha=0.6)
                    self.ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
                    self.ax.set_xlabel(x_column)
                    self.ax.set_ylabel(y_column)
                else:
                    QMessageBox.warning(self, "Warning", "Scatter plot requires numeric columns")
                    return
                    
            elif chart_type == "Box Plot":
                if self.df[x_column].dtype in ['int64', 'float64']:
                    self.df.boxplot(column=x_column, ax=self.ax)
                    self.ax.set_title(f"Box Plot of {x_column}")
                    self.ax.set_ylabel(x_column)
                else:
                    QMessageBox.warning(self, "Warning", "Box plot requires numeric columns")
                    return
                    
            elif chart_type == "Line Chart":
                if self.df[x_column].dtype in ['int64', 'float64']:
                    if y_column and self.df[y_column].dtype in ['int64', 'float64']:
                        self.df.plot(x=x_column, y=y_column, ax=self.ax, marker='o')
                        self.ax.set_title(f"Line Chart: {x_column} vs {y_column}")
                        self.ax.set_xlabel(x_column)
                        self.ax.set_ylabel(y_column)
                    else:
                        self.df[x_column].plot(ax=self.ax, marker='o')
                        self.ax.set_title(f"Line Chart of {x_column}")
                        self.ax.set_xlabel("Index")
                        self.ax.set_ylabel(x_column)
                else:
                    QMessageBox.warning(self, "Warning", "Line chart requires numeric columns")
                    return
                    
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            self.statusBar().showMessage(f"{chart_type} created successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create chart: {str(e)}")
            
    def export_data(self):
        """Export data to various formats"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        dialog = ExportDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            export_info = dialog.get_export_info()
            
            if export_info['file_path'] == "No file selected":
                QMessageBox.warning(self, "Warning", "Please select a file path")
                return
                
            try:
                success = export_data(self.df, export_info['file_path'], export_info['format'])
                
                if success:
                    QMessageBox.information(self, "Success", f"Data exported successfully to {export_info['file_path']}")
                    self.statusBar().showMessage("Data exported successfully")
                else:
                    QMessageBox.critical(self, "Error", "Failed to export data")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
