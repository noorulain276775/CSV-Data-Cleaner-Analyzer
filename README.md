# CSV Data Cleaner Analyzer

A comprehensive PyQt5-based desktop application for cleaning, analyzing, and visualizing CSV data using Pandas and advanced data processing techniques.

## Features

### Basic Version
- **CSV Loading**: Load and view CSV files with automatic data detection
- **Data Viewing**: Interactive table view with data information (rows, columns, memory usage)
- **Basic Cleaning**: Remove null values with configurable strategies
- **Data Filtering**: Apply filters using various operators (==, !=, >, <, >=, <=, contains)
- **Data Analysis**: Generate summary statistics and column-specific analysis
- **Data Visualization**: Create charts including histograms, bar charts, scatter plots, box plots, and line charts
- **Data Reset**: Reset to original data after cleaning operations

### Enhanced Version (Additional Features)
- **Advanced Cleaning**: 
  - Duplicate removal with multiple strategies
  - Missing value handling (mean, median, mode, forward/backward fill)
  - Column name cleaning and optimization
  - Data type detection and suggestions
- **Data Quality Reporting**: Comprehensive data quality metrics and analysis
- **Outlier Detection**: Identify outliers using IQR and Z-score methods
- **Data Export**: Export cleaned data to multiple formats (CSV, Excel, JSON, Parquet)
- **Enhanced UI**: Additional tabs and advanced features

## Screenshots

The application features a modern, tabbed interface with:
- **Data View Tab**: Shows loaded data with statistics
- **Data Cleaning Tab**: Tools for data cleaning and filtering
- **Data Analysis Tab**: Statistical analysis and column insights
- **Visualization Tab**: Interactive charts and graphs
- **Advanced Features Tab**: Quality reporting and outlier detection (enhanced version only)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone or download this repository
2. Navigate to the project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

#### Option 1: Version Selector (Recommended)
```bash
python app_launcher.py
```
This will show a window where you can choose between basic and enhanced versions.

#### Option 2: Direct Version Selection
```bash
# Run basic version
python app_launcher.py basic

# Run enhanced version  
python app_launcher.py enhanced
```

#### Option 3: Legacy Basic Version
```bash
python main.py
```

### Basic Workflow
1. **Load Data**: Click "Browse" to select a CSV file
2. **Explore Data**: View data in the Data View tab
3. **Clean Data**: Use the Data Cleaning tab to remove nulls, apply filters, etc.
4. **Analyze Data**: Generate summaries and analyze specific columns
5. **Visualize Data**: Create charts and graphs in the Visualization tab
6. **Export Results**: Save cleaned data in your preferred format

### Advanced Features (Enhanced Version)
- **Data Quality Report**: Click "Generate Quality Report" to get comprehensive data insights
- **Outlier Detection**: Select a numeric column and detect outliers using IQR or Z-score methods
- **Advanced Cleaning**: Use the "Advanced Cleaning Options" button for comprehensive data cleaning
- **Multiple Export Formats**: Export data to CSV, Excel, JSON, or Parquet formats

## Sample Data

The project includes `sample_data.csv` with sample employee data for testing:
- Name, Age, City, Salary, Department, Experience
- 20 sample records with various data types

## File Structure

```
CSV-Data-Cleaner-Analyzer/
├── main.py                      # Basic version entry point
├── main_screen.py              # Basic version GUI
├── app_launcher.py             # Main launcher with version selector
├── enhanced_gui.py             # Enhanced version GUI implementation
├── utils.py                    # Utility functions for data processing
├── requirements.txt            # Python dependencies
├── sample_data.csv            # Sample data for testing
└── README.md                  # This file
```

## Dependencies

- **PyQt5**: GUI framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **openpyxl**: Excel file support
- **xlrd**: Excel file reading

## Key Features Explained

### Data Cleaning
- **Null Removal**: Remove rows with missing values using 'any' or 'all' strategies
- **Filtering**: Apply complex filters with multiple operators
- **Duplicate Handling**: Remove duplicate rows with configurable strategies
- **Missing Value Imputation**: Fill missing values using statistical methods

### Data Analysis
- **Summary Statistics**: Comprehensive statistical overview of numeric columns
- **Column Analysis**: Detailed analysis of individual columns including data types, unique values, and statistics
- **Data Quality Metrics**: Memory usage, duplicate detection, missing value analysis

### Visualization
- **Histograms**: Distribution analysis of numeric columns
- **Bar Charts**: Frequency analysis of categorical data
- **Scatter Plots**: Relationship analysis between numeric columns
- **Box Plots**: Statistical distribution and outlier visualization
- **Line Charts**: Trend analysis and time series visualization

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **PyQt5 Issues**: On some systems, you may need to install system-level Qt dependencies
3. **Memory Issues**: Large CSV files (>100MB) may cause performance issues; consider using data sampling

### Performance Tips
- The application limits table display to 1000 rows for performance
- Use filters to reduce data size before analysis
- Close unnecessary tabs to free up memory

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the documentation
- Submitting pull requests

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Open an issue on the project repository

---

**Note**: This application is designed for educational and professional data analysis purposes. Always backup your original data before performing cleaning operations.
