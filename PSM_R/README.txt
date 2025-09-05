# README for Baseline Table and Propensity Score Matching

## Overview
This R script performs baseline table creation and propensity score matching using the `tableone` and `MatchIt` libraries. It processes a dataset to group BMI values, match cases and controls, and generate summary statistics.

## Functionality
1. **Data Reading**: The script reads a CSV file containing baseline data.
2. **BMI Grouping**: It categorizes the BMI variable into different types (Underweight, Normal, Overweight, Obese).
3. **Table Creation**: It creates a baseline table summarizing the characteristics of the dataset.
4. **Propensity Score Matching**: The script matches cases and controls based on specified covariates using nearest neighbor matching.
5. **Output**: The results, including matched data and summary tables, are saved to CSV files.

## Key Components
- **Data Input**: The script reads data from `data_baseline_english.csv`.
- **BMI Categorization**: The BMI variable is categorized using the `within` function.
- **TableOne Creation**: The `CreateTableOne` function is used to create summary tables.
- **Matching**: The `matchit` function from the `MatchIt` library is used for propensity score matching.
- **Output Files**: The script generates several output files:
  - `data_result_matched.csv`: Contains the matched dataset.
  - `result_matched.csv`: Merged results of matched quantitative data.
  - `result_before.csv`: Summary statistics before matching.
  - `result_after.csv`: Summary statistics after matching.

## Requirements
- R (version 3.5 or higher)
- Required packages: `tableone`, `MatchIt`

## Usage
1. Ensure the required packages are installed.
2. Set the correct path for the input CSV file.
3. Run the script to perform the analysis and generate output files.

## Note
Make sure to check the output files for the results of the analysis and ensure that the data is correctly formatted before running the script.