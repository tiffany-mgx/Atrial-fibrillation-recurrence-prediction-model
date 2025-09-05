# README for Code Upload Project

## Overview
This project contains code for analyzing radiomics features in medical imaging, specifically focusing on atrial fibrillation. The code is structured to facilitate data processing, model training, feature selection, and evaluation using various machine learning techniques. The goal is to enhance the understanding of imaging biomarkers and their relationship with clinical outcomes.

## Project Structure
The project is organized into the following directories and files:

- **dataset/**: Contains scripts for generating JSON files from clinical data and images.
  - `generate_json_clinical_cohort.py`: Script to generate JSON files for clinical cohorts, which includes patient information and imaging data.
  - `generate_json_train.py`: Script to generate training data in JSON format, which is used for Swinunetr model training.
  - `images/`: Directory for storing image files in NIfTI format.
  - `labels/`: Directory for storing corresponding label files for the images.

- **PSM_R/**: Contains scripts for propensity score matching and analysis.
  - `codes.R`: R script for creating baseline tables and performing propensity score matching, which helps in balancing the characteristics of case and control groups.

- **radiomics_extraction/**: Contains scripts for extracting radiomics features from medical images.
  - `radiomics_feature_extract_from_nii.py`: Python script for extracting features from NIfTI images using the `pyradiomics` library.

- **radiomics_selectBest_area_analysis.py**: Python script for training radiomics models using k-fold cross-validation and selecting the best features based on performance metrics.

- **radiomics_selectBest_area_SHAP_analysis.py**: Python script for analyzing the selected features and generating SHAP values for model interpretability, allowing for insights into feature importance.

- **transfer_learning_in_clinical_cohort.py**: Python script for loading pre-trained models and performing intelligent segmentation of case and control group images, utilizing transfer learning techniques to improve segmentation accuracy.

## Running the Code
To run the code in this project, follow these steps:

1. **Set Up Environment**: Ensure you have Python 3.x and the required libraries installed. You may need to install the following packages:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `xgboost`
   - `lightgbm`
   - `torch`
   - `matplotlib`
   - `seaborn`
   - `shap`
   - `monai` (for medical imaging tasks)

   You can install the required packages using pip:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn shap monai
   ```

2. **Prepare Data**: Place your medical imaging data in the appropriate directories as specified in the scripts. Ensure that the data is formatted correctly, with images in NIfTI format and labels corresponding to the images.

3. **Run Scripts**: Execute the scripts in the following order:
   - Generate JSON files using the scripts in the `dataset/` directory. This step is crucial for preparing the data for model training.
   - Perform transfer learning and segmentation using `transfer_learning_in_clinical_cohort.py`. This step will utilize pre-trained models to segment images intelligently.
   - Extract radiomics features using `radiomics_feature_extract_from_nii.py`. This will create a dataset of features that can be used for model training.
   - Train models and select features using `radiomics_selectBest_area_analysis.py`. This script will perform k-fold cross-validation to evaluate model performance.
   - Analyze features and generate SHAP values using `radiomics_selectBest_area_SHAP_analysis.py`. This will provide insights into which features are most important for the model's predictions.

4. **Output**: The results will be saved in the specified output directories, including CSV files for model evaluation and SHAP values. You will find:
   - `result_output_df_radiomics.csv`: Quantitative results from the ten-fold cross-validation.
   - `result_output_plot_all_df_radiomics.csv`: Summary of all cross-validation results.
   - `SHAP_values_df.csv`: SHAP values for feature importance.
   - Segmented images saved in the specified output paths.

## Important Notes
- **Data Privacy**: Ensure that any patient data used in this project complies with relevant data protection regulations (e.g., HIPAA, GDPR).
- **Parameter Tuning**: The scripts contain TODO comments where you may need to adjust parameters based on your specific dataset and research questions.
- **Model Selection**: The project includes various models (e.g., XGBoost, Random Forest, SVM) for training. You can modify the `MODELS` dictionary in the training scripts to include or exclude models as needed.

## Conclusion
This project aims to provide a comprehensive framework for analyzing radiomics features in medical imaging, facilitating research in atrial fibrillation and related conditions. By following the outlined steps, you can replicate the analysis and adapt the code for your specific research needs. For any questions or issues, please refer to the documentation or contact the author.

## Contact
For further inquiries, please reach out to Guoxiang Ma.