'''
    Artificial intelligence methods were employed to train the various structures of the heart.
'''

import os
import shutil
import tempfile
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

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
    ResizeWithPadOrCropd,
    Resized
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

print_config()

# Setup data directory
# directory = os.environ.get("MONAI_DATA_DIRECTORY")
directory = "./model_saved"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
cls = 2  #  Model output classes
patch_sizeX = 96  # 96
patch_sizeZ = 96  # patch_size depth must be divisible by image_sizeZ
image_sizeX = 512
image_sizeY = 512
image_sizeZ = 512  # Average thickness of all cases is 218
slicer_index = 150
case_num = 2  # Which case number
dir_list = ['1_left_atrium', '2_left_aurcle', '3_left_ventriculus_snister', '4_right_atrium',
            '5_right_ventriculus_dexter', '6_pulmonary_vein']
area_name = dir_list[3]  # Save different names for different models

# Setup transforms for training and validation
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Resized(keys=["image", "label"], spatial_size=(image_sizeX, image_sizeY, image_sizeZ)),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(image_sizeX, image_sizeY, image_sizeZ)),  # Modify
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(patch_sizeX, patch_sizeX, patch_sizeZ),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        )
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Resized(keys=["image", "label"], spatial_size=(image_sizeX, image_sizeY, image_sizeZ))
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(image_sizeX, image_sizeY, image_sizeZ)),  # Modify
    ]
)

data_dir = "./dataset/"  # Root directory
split_json = "dataset_train_" + str(area_name) + ".json"
datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")  # Read json file
val_files = load_decathlon_datalist(datasets, True, "validation")

train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=0,
)
# Batch size data settings
batch_size = 1
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# slice_map = {
#     "9.nii.gz": 250  # Which slice number, must be data from the validation set
# }
# case_num = 0  # Which case number
img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
print(img_name)
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")  # image shape: torch.Size([1, 512, 512, 512]), label shape: torch.Size([1, 512, 512, 512])
plt.figure("image", (18, 6))
plt.figure("image")
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, img_shape[3]-slicer_index].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, img_shape[3]-slicer_index].detach().cpu())
plt.show()


# Create model loss optimizer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model training device: ", device)

model = SwinUNETR(
    img_size=(patch_sizeX, patch_sizeX, patch_sizeZ),
    in_channels=1,
    out_channels=cls,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# Initialize swin unetr encoder from self-supervised pre-trained weights
# https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
weight = torch.load("./pretrained_model/model_swinvit.pt")
model.load_from(weights=weight)
print("Using pretrained self-supervised Swin UNETR backbone weights!")


loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Execute a typical pytorch training process
def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (patch_sizeX, patch_sizeX, patch_sizeZ), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_" + str(area_name) + str(dice_val_best) + ".pth"))
                print(
                    "Model Was Saved! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 10000
eval_num = 500
post_label = AsDiscrete(to_onehot=cls)
post_pred = AsDiscrete(argmax=True, to_onehot=cls)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_" + str(area_name) + str(dice_val_best) + ".pth")))

print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
with open('./result/list_loss_' + str(area_name) + '.pkl', 'wb') as f:
    pickle.dump([x, y], f)

x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
with open('./result/list_metric_value_' + str(area_name) + '.pkl', 'wb') as f:
    pickle.dump([x, y], f)


# Visualization of segmentation results
# Check best model output with the input image and label
case_num = 0
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_nosieze" + str(dice_val_best) + ".pth")))
model.eval()
with torch.no_grad():
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(val_inputs, (patch_sizeX, patch_sizeX, patch_sizeZ), 4, model, overlap=0.8)
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, img_shape[3]-slicer_index], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, img_shape[3]-slicer_index])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, img_shape[3]-slicer_index])
    plt.show()
