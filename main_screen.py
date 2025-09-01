#!/usr/bin/env python3
"""
CSV Data Cleaner Analyzer
Main screen with GUI components
"""

import os
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QComboBox, QLineEdit, QTextEdit,
                             QGroupBox, QGridLayout, QMessageBox, QTabWidget,
                             QProgressBar, QSpinBox, QCheckBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

class DataProcessor(QThread):
    """Thread for processing data operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, operation, data, **kwargs):
        super().__init__()
        self.operation = operation
        self.data = data
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.operation == "remove_nulls":
                result = self.data.dropna(**self.kwargs)
            elif self.operation == "filter":
                column = self.kwargs.get('column')
                value = self.kwargs.get('value')
                operator = self.kwargs.get('operator', '==')
                if operator == '==':
                    result = self.data[self.data[column] == value]
                elif operator == '!=':
                    result = self.data[self.data[column] != value]
                elif operator == '>':
                    result = self.data[self.data[column] > value]
                elif operator == '<':
                    result = self.data[self.data[column] < value]
                elif operator == '>=':
                    result = self.data[self.data[column] >= value]
                elif operator == '<=':
                    result = self.data[self.data[column] <= value]
                elif operator == 'contains':
                    result = self.data[self.data[column].str.contains(str(value), na=False)]
            elif self.operation == "summarize":
                result = self.data.describe()
            else:
                result = self.data
                
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """Main window for the CSV Data Cleaner Analyzer"""
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.original_df = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("CSV Data Cleaner Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create title
        title_label = QLabel("CSV Data Cleaner Analyzer")
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
        
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(load_button)
        
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
        
        info_layout.addWidget(self.row_count_label, 0, 0)
        info_layout.addWidget(self.col_count_label, 0, 1)
        info_layout.addWidget(self.memory_label, 0, 2)
        
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
        
        # Remove nulls section
        nulls_group = QGroupBox("Remove Null Values")
        nulls_layout = QGridLayout()
        
        self.nulls_combo = QComboBox()
        self.nulls_combo.addItems(["any", "all"])
        self.nulls_combo.setCurrentText("any")
        
        remove_nulls_btn = QPushButton("Remove Nulls")
        remove_nulls_btn.clicked.connect(self.remove_nulls)
        
        nulls_layout.addWidget(QLabel("Strategy:"), 0, 0)
        nulls_layout.addWidget(self.nulls_combo, 0, 1)
        nulls_layout.addWidget(remove_nulls_btn, 0, 2)
        
        nulls_group.setLayout(nulls_layout)
        layout.addWidget(nulls_group)
        
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
