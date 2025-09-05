
# Data reading
import glob
import os
path_originalData = "your dataset path"

area_num = 1  # Generate a separate json file for each area, adjust this value from 0 to 5!
dir_list = ['1_left_atrium', '2_left_aurcle', '3_left_ventriculus_snister', '4_right_atrium',
            '5_right_ventriculus_dexter', '6_pulmonary_vein']

image_path = path_originalData + '\\case_and_control_output\\' + str(dir_list[area_num])  # Model output image path
label_path = path_originalData + '\\case_and_control_output_label\\' + str(dir_list[area_num])  # Model output label path

train_image_list = glob.glob(image_path + "\\*.nii.gz")
print(len(train_image_list))
# Verify if the data labels match
imageList = []
for i, cur_img in enumerate(train_image_list):
    imageList.append(cur_img.split('\\')[-1])
print("Number of case data: ", len(imageList))

train_image = imageList
print("train size: {} ".format(len(train_image)))

# Build json file
import json
from collections import OrderedDict

json_dict = OrderedDict()
json_dict['name'] = "heart"
json_dict['tensorImageSize'] = "3D"
json_dict['release'] = "0.0"
json_dict['modality'] = {"0": "CTA"}

json_dict['labels'] = {
    "0": "Background",
    "1": "segmentation1"
}

json_dict['numTraining'] = len(train_image)

json_dict['training'] = []
for idx in range(len(train_image)):
    json_dict['training'].append(
        {'image': image_path + "\\" + train_image[idx],
         'label':  label_path + "\\" + train_image[idx]
         })

with open("./dataset/dataset_transfer_clinical_cohort_" + str(dir_list[area_num]) + ".json", 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)

