
# Data reading
import glob
import os
path_originalData = "your dataset path"
area_num = 0  # Switch to different areas!!
dir_list = ['1_left_atrium', '2_left_aurcle', '3_left_ventriculus_snister', '4_right_atrium',
            '5_right_ventriculus_dexter', '6_pulmonary_vein']
train_image_list = glob.glob(path_originalData + "\\heart_images_nii\\*.nii.gz")
train_label_list = glob.glob(path_originalData + "\\heart_multi_organs_labels_nii\\" + str(dir_list[area_num]) + "\\*.nii.gz")

# Verify if the data labels match
imageList = []
for i, cur_img in enumerate(train_image_list):
    cur_img_new = cur_img.replace('\\heart_images_nii', '\\heart_multi_organs_labels_nii\\' + str(dir_list[area_num]))
    if cur_img_new not in train_label_list:
        print("Data matching exception.....")
        exit(0)
    else:
        imageList.append(cur_img.split('\\')[-1])
print("Number of case data: ", len(imageList))

val_size = 10
train_image = imageList[:-val_size]
train_label = imageList[:-val_size]
validation_image = imageList[-val_size:]
validation_label = imageList[-val_size:]
print("train size: {} ,valid size: {}".format(len(train_image), len(validation_image)))

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
label_path = path_originalData + '\\heart_multi_organs_labels_nii\\' + str(dir_list[area_num])
for idx in range(len(train_image)):
    json_dict['training'].append(
        {'image': path_originalData + "\\heart_images_nii\\{}".format(train_image[idx]),
         'label':  label_path + "\\" + train_label[idx]
         })

json_dict['validation'] = []
label_path = path_originalData + '\\heart_multi_organs_labels_nii\\' + str(dir_list[area_num])
for idx in range(len(validation_image)):
    json_dict['validation'].append(
        {
         'image': path_originalData + "\\heart_images_nii\\{}".format(validation_image[idx]),
         'label': label_path + "\\" + validation_label[idx]
        })

with open("./dataset/dataset_train_" + str(dir_list[area_num]) + ".json", 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)

