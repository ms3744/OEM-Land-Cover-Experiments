import os
import time
import warnings
import numpy as np
import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import oem
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Path to the OpenEarthMap directory
OEM_DATA_DIR = "../OpenEarthMap_Mini/"

# Training and validation file list
TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

IMG_SIZE = 512
N_CLASSES = 9
LR = 0.0001
WEIGHT_DECAY = 0.000001
BATCH_SIZE = 16
NUM_EPOCHS = 230
DEVICE = "cuda"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# VARIABLES TO TEST:
batch_sizes = [4, 8, 16]
backbones = ['res2net101_26w_4s', 'res2next50',
             'seresnet152d', 'swsl_resnet18']
#  'efficientnet_b1_pruned', 'efficientnet_b0', 'resnetv2_101x1_bitm']
optimizers = ['adam', 'sgd']
criterions = ['CE', 'Focal', 'Jaccard', 'Dice', 'MCC', 'CE+', 'MCC+']


def run_one_model(train_data, val_data, batch_size, backbone, optim, crit, epochs=NUM_EPOCHS, train_existing=False):

    # Loading dataset
    print('Loading data')
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    # Load model
    network = oem.networks.UNetFormer(
        in_channels=3, n_classes=N_CLASSES, backbone_name=backbone)

    # Load optimizer
    # IMP: models tend to overfit, so adding weight decay
    print('Loading Optimizer')
    if optim == 'adam':
        optimizer = torch.optim.Adam(
            network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(
            network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Load criterion
    # 'CE', 'Focal', 'Jaccard', 'Dice', 'MCC', 'CE+', 'MCC+'
    print('Loading Criterion')
    floats = False
    if crit == 'CE':
        criterion = [oem.losses.CEWithLogitsLoss()]
    elif crit == 'Focal':
        criterion = [oem.losses.FocalLoss()]
        floats = True
    elif crit == 'Jaccard':
        criterion = [oem.losses.JaccardLoss()]
    elif crit == 'Dice':
        criterion = [oem.losses.DiceLoss()]
    elif crit == 'MCC':
        criterion = [oem.losses.MCCLoss()]
    elif crit == 'CE+':
        criterion = [oem.losses.CEWithLogitsLoss(
        ), oem.losses.FocalLoss(), oem.losses.MCCLoss()]
    else:
        criterion = [oem.losses.CEWithLogitsLoss(), oem.losses.MCCLoss()]

    # Emptying cache
    torch.cuda.empty_cache()

    start = time.time()

    max_score = 0

    ##########################################################################################    ##########################################################################################
    model_path = OUTPUT_DIR+"/"+network.name+"_" + \
        str(batch_size)+"_"+optim+"_"+crit+"_mini.pth"

    model_exists = False
    if os.path.isfile(model_path):
        print(model_path)
        network_loaded = oem.utils.load_checkpoint(
            network, network.name+"_" + str(batch_size)+"_"+optim+"_"+crit+"_mini.pth", "../outputs/")
        model_exists = True

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}")

        if os.path.isfile(model_path) and model_exists and not train_existing:
            # print(model_path)
            # network_loaded = oem.utils.load_checkpoint(
            #     network, network.name+"_" + str(batch_size)+"_"+optim+"_"+crit+".pth", "../outputs/")
            valid_logs = oem.runners.valid_epoch(
                model=network_loaded,
                criterion=criterion,
                dataloader=val_data_loader,
                device=DEVICE,
                floats=floats
            )
            return valid_logs["Score"], network_loaded.name
        else:
            print("NOT FOUND")
            print(model_path)
            train_logs = oem.runners.train_epoch(
                model=network,
                optimizer=optimizer,
                criterion=criterion,
                dataloader=train_data_loader,
                device=DEVICE,
                floats=floats
            )

            valid_logs = oem.runners.valid_epoch(
                model=network,
                criterion=criterion,
                dataloader=val_data_loader,
                device=DEVICE,
                floats=floats
            )

        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name=network.name+"_" +
                str(batch_size)+"_"+optim+"_"+crit+"_mini.pth",
                output_dir=OUTPUT_DIR,
            )

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
    return max_score, network.name


def main():
    img_paths = [f for f in Path(OEM_DATA_DIR).rglob(
        "*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in img_paths if f.name in np.loadtxt(
        TRAIN_LIST, dtype=str)]
    val_fns = [str(f)
               for f in img_paths if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    print("Total samples      :", len(img_paths))
    print("Training samples   :", len(train_fns))
    print("Validation samples :", len(val_fns))

    train_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Rotate(),
            oem.transforms.Crop(IMG_SIZE),
        ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(IMG_SIZE),
        ],

    )

    train_data = oem.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    data_list = []
    for batch in batch_sizes:
        for backbone in backbones:
            for optimizer in optimizers:
                for criterion in criterions:
                    best_score, name = run_one_model(
                        train_data, val_data, batch, backbone, optimizer, criterion)

                    data_point = {
                        'Model': name,
                        'Batch_size': batch,
                        'Optimizer': optimizer,
                        'Criterion': criterion,
                        'Best Score': best_score
                    }

                    data_list.append(data_point)

    results_df = pd.DataFrame(data_list)
    results_df.to_csv('unetformer_results_seresnet.csv', index=False)


if __name__ == '__main__':
    main()
