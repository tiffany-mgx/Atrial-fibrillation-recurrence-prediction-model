'''
	Imaging Biomarker Case Program
	Author: Ma Guoxiang
	Function: Extract imaging biomarker features and output them to a CSV file.
'''
import os
import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
import time
import pandas as pd
import warnings
import glob
import json


warnings.filterwarnings("ignore")
root_path = '\your dataset path'


area_num = 0  # TODO: Switch to different areas!!
dir_list = ['1_left_atrium', '2_left_aurcle', '3_left_ventriculus_snister', '4_right_atrium',
            '5_right_ventriculus_dexter', '6_pulmonary_vein']


# Read the images for radiomics feature extraction from the json file
def image_process():
    with open("../dataset/dataset_transfer_clinical_cohort_" + dir_list[0] + ".json", encoding='utf-8') as file:
        data = json.load(file)
    return data['training']


# Radiomics feature extraction
def extract_radiomics_feature(image, mask):
    settings = {}
    settings['binWidth'] = 25
    settings['Interpolator'] = sitk.sitkBSpline
    settings['resampledPixelSpacing'] = [2, 2, 2]
    settings['voxelArrayShift'] = 1000
    settings['normalize'] = True
    settings['normalizeScale'] = 100

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # extractor.enableImageTypeByName('Wavelet')
    # extractor.enableImageTypeByName('LoG', [10.0])
    extractor.enableAllFeatures()
    extractor.enableFeaturesByName(
        firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean',
                    'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
                    'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'],
        shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2',
               'Sphericity', 'SphericalDisproportion', 'Maximum3DDiameter', 'Maximum2DDiameterSlice',
               'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
               'LeastAxisLength', 'Elongation', 'Flatness'],
        shape2D=['MeshSurface', 'PixelSurface', 'Perimeter', 'PerimeterSurfaceRatio', 'Sphericity',
                 'SphericalDisproportion',
                 'MaximumDiameter', 'MajorAxisLength', 'MinorAxisLength', 'Elongation'],
        glcm=['Autocorrelation', 'JointAverage', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast',
              'Correlation', 'DifferenceAverage', 'DifferenceEntropy', 'DifferenceVariance',
              'JointEnergy', 'JointEntropy', 'Imc1', 'Imc2', 'Idm', 'MCC', 'Idmn',
              'Id', 'Idn', 'InverseVariance', 'MaximumProbability', 'SumAverage', 'SumEntropy',
              'SumSquares'], 
    )

    result = extractor.execute(image, mask)  # Extract features

    result_copy = result.copy()
    # TODO: Print the extracted radiomics features
    for featureName in result_copy.keys():
        if str(featureName).startswith('diagnostics'):
            del result[featureName]   # Delete irrelevant radiomics features
    print("Number of radiomics features: ", len(result.values()))
    return result


if __name__ == '__main__':
    start_time = time.time()
    # Read data directory
    data_list_json = image_process()

    radiomics_features_list, image_names_list = list(), list()

    # Radiomics feature extraction
    for img_name in data_list_json:
        img_path = img_name['image']
        img_mask_name = img_name['label']
        image_names_list.append(str(img_name['image']).split('/')[-1])
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Current feature extraction image name: ", img_name['image'])
        try:
            cur_radiomics_feature = extract_radiomics_feature(img_path, img_mask_name)    # Extract radiomics features
            radiomics_features_list.append(cur_radiomics_feature.values())
            print(cur_radiomics_feature.values())
        except:
            print("An exception occurred while extracting radiomics features from the current image...")
            continue

    feature_output = pd.DataFrame(columns=cur_radiomics_feature.keys(), data=radiomics_features_list)

    feature_output['image_name'] = image_names_list
    print("Radiomics feature dimension: ", feature_output.shape)  # Radiomics feature dimension:  (311, 112)

    end_time = time.time()
    date_time = time.strftime("%Y%m%d_%H_%M_%S", time.gmtime(end_time))
    feature_output.to_csv("../dataset/result_radiomics_feature_" + dir_list[area_num] + str(date_time) + ".csv")
    print("Total running time of the program: %.2f seconds" % (end_time-start_time))
