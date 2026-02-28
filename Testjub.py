import torch
<<<<<<< HEAD
<<<<<<< HEAD
from monai.networks.nets import UNETR
=======
from monai.networks.nets import UNet
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
=======
from monai.networks.nets import UNet
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, EnsureTyped, SaveImaged
import os
from monai.inferers import sliding_window_inference


<<<<<<< HEAD
<<<<<<< HEAD
file_path = "/lustrefs/disk/home/climpana/wat/test_ids.txt"
=======
file_path = "/lustrefs/disk/home/sngamvil/test_ids.txt"
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
=======
file_path = "/lustrefs/disk/home/sngamvil/test_ids.txt"
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
with open(file_path, "r") as f:
  test_ids = [line.strip() for line in f if line.strip()]
  
images_dir = "/lustrefs/disk/project/lt200431-ddmmss/wat/datasets--alexanderdann--CTSpine1K/snapshots/9b454add169b94f2c322ad6f08b66823975e8dbd/raw_data/volumes/COLONOG/"

<<<<<<< HEAD
<<<<<<< HEAD
num_classes = 7
model = UNETR(
    in_channels = 1,
    out_channels = num_classes,
    img_size = [128, 128, 128],
    feature_size = 48,
    hidden_size = 768,
    mlp_dim = 3072,
    num_heads = 12,
).cuda()

model_path = "/lustrefs/disk/home/climpana/wat/MySpineSAM3/checkpoints/best_model.pth"
checkpoint = torch.load(model_path, map_location="cuda:0", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
=======
=======
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
num_classes = 19
model = UNet(spatial_dims = 3,
            in_channels = 1,
            out_channels = num_classes,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2, 
).cuda()

model_path = "/lustrefs/disk/home/sngamvil/sam3/best_4gpu_model.pt"
state_dict = torch.load(model_path, map_location="cuda:0")
model.load_state_dict(state_dict)
<<<<<<< HEAD
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
=======
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
model.eval()

transform = Compose([
    LoadImaged(keys = ["image"]),
    EnsureChannelFirstd(keys = ["image"]),
    Orientationd(keys = ["image"], axcodes = "RAS"),
    Spacingd(keys = ["image"],
        pixdim = (1.0, 1.0, 1.0),
        mode = "bilinear"
    ),
    ScaleIntensityRanged(keys = ["image"],
<<<<<<< HEAD
<<<<<<< HEAD
        a_min = -100, a_max = 1000,
=======
        a_min = -1000, a_max = 1000,
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
=======
        a_min = -1000, a_max = 1000,
>>>>>>> e32fd73e28d2302f00d10893189933519a467e75
        b_min = 0.0, b_max = 1.0,
        clip = True
    ),
    SpatialPadd(keys = ["image"],
        spatial_size = None,
        divisible = (16, 16, 16)
    ),
    EnsureTyped(keys = ["image"],
                data_type = "tensor",
                track_meta = True
    ),
])

saver = SaveImaged(keys="pred",
    output_dir = "./predictions",
    output_postfix = "seg",
    output_ext = ".nii.gz",
    resample = False
)

for ids in test_ids:
    image_path = os.path.join(images_dir, ids + ".nii.gz")

    data = {"image": image_path}
    data = transform(data)
    image = data["image"].unsqueeze(0).cuda()

    with torch.no_grad():
        logits = sliding_window_inference(inputs = image,
                                          roi_size = (128, 128, 128),  
                                          sw_batch_size = 1,
                                          predictor = model,
                                          overlap = 0.5,
                                          mode = "gaussian"
        )
        pred = torch.argmax(logits, dim = 1)

    data["pred"] = pred
    saver(data)

    print(f"Saved segmentation for {ids}")
  
  