# README for Imaging Biomarker Case Program

## Overview
This program is designed to extract imaging biomarker features from medical images and output them to a CSV file. It utilizes the `radiomics` library for feature extraction and `SimpleITK` for image processing.

## Author
- Ma Guoxiang

## Functionality
1. **Image Processing**: The program reads images and their corresponding labels from a JSON file.
2. **Feature Extraction**: It extracts various radiomics features from the images using specified settings.
3. **Output**: The extracted features are saved to a CSV file for further analysis.

## Key Components
- **image_process()**: Reads the JSON file containing image paths and labels.
- **extract_radiomics_feature(image, mask)**: Extracts radiomics features from the given image and mask.
- **Main Execution**: The main block of the code orchestrates the reading of data, feature extraction, and saving results.

## Requirements
- Python 3.x
- Required libraries: `numpy`, `pandas`, `SimpleITK`, `radiomics`, `json`, `glob`, `warnings`, `time`

## Usage
1. Set the `root_path` variable to the path of your dataset.
2. Ensure the JSON files are correctly formatted and located in the specified directory.
3. Run the script to extract features and generate the output CSV file.

## Note
Make sure to handle exceptions properly to avoid interruptions during feature extraction.