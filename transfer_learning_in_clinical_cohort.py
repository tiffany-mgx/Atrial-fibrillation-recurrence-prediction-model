
'''
    Load the pre-trained model and intelligently segment the images of the case group and control group
'''
import os
import tempfile
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.networks.nets import SwinUNETR

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ResizeWithPadOrCropd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

print_config()

# Setup data directory
# directory = os.environ.get("MONAI_DATA_DIRECTORY")
directory = "./model_saved"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
cls = 2  # Model output classes
patch_sizeX = 96
patch_sizeZ = 96
image_sizeX = 512
image_sizeY = 512
image_sizeZ = 512
best_model_name = "best model name"  # Best model
area_num = 0
dir_list = ['1_left_atrium', '2_left_aurcle', '3_left_ventriculus_snister', '4_right_atrium',
            '5_right_ventriculus_dexter', '6_pulmonary_vein']

# Create model loss optimizer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data preparation
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),  # No rotation transformation for the test set
        Spacingd(  # Resolution adjustment, adding slices will change the number of layers!
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        # ResizeWithPadOrCropd(keys=["image"], spatial_size=(image_sizeX, image_sizeY, image_sizeZ)),  # Modify
    ]
)

data_dir = "./dataset/"
split_json = "dataset_transfer_clinical_cohort_" + dir_list[area_num] + ".json"

datasets = data_dir + split_json
test_files = load_decathlon_datalist(datasets, True, "training")
test_ds = CacheDataset(
    data=test_files,
    transform=test_transforms,
    cache_num=6,
    cache_rate=1.0,
    num_workers=0
    )

# 2. Model loading
model = SwinUNETR(
    img_size=(patch_sizeX, patch_sizeX, patch_sizeZ),
    in_channels=1,
    out_channels=cls,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# Check best model output with the input image and label
case_num = 0
model.load_state_dict(torch.load(os.path.join(root_dir, best_model_name)))
model.eval()
output_path = ".\\heart_cohort_nii\\case_and_control_output\\" + dir_list[area_num]
output_label_path = ".\\heart_cohort_nii\\case_and_control_output_label\\" + dir_list[area_num]
for case_num in range(len(test_files)):
    print("Current predicted image: ", test_files[case_num])
    with torch.no_grad():
        img_name = os.path.split(test_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = test_ds[case_num]["image"]
        # print(img.shape)  # torch.Size([1, 189, 188, 81])
        test_inputs = torch.unsqueeze(img, 1).cuda()  # torch.Size([1, 1, 189, 188, 81])
        print(test_inputs.shape)
        test_outputs = sliding_window_inference(test_inputs, (patch_sizeX, patch_sizeX, patch_sizeZ), 4, model, overlap=0.8)  # torch.Size([1, 2, 512, 512, 61])
        current_img_pred = torch.argmax(test_outputs, dim=1).detach().cpu()[0].numpy()

        out_img = nib.Nifti1Image(img.squeeze().numpy().astype(np.float32), None)
        out_pred = nib.Nifti1Image(current_img_pred.astype(np.float32), None)
        print("out_img size: ", out_img.shape)
        print("out_pred size: ", out_pred.shape)
        nib.save(out_img,  output_path + "\\" + img_name)
        nib.save(out_pred, output_label_path + "\\" + img_name)
