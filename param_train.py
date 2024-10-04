import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model_archs.SingleStream3D import SingleStream3D
from model_archs.AdvancedConv2DModel import AdvancedConv2DModel
from model_archs.AdvancedConv2DModel1 import AdvancedConv2DModel1
from model_archs.AdvancedConv2DModelV2 import AdvancedConv2DModelV2
from model_archs.AdvancedConv2DModelV3 import AdvancedConv2DModelV3
from model_archs.VerticalStackConv2DModel import VerticalStackConv2DModel
from model_archs.VerticalStackConv2DModelV1 import VerticalStackConv2DModelV1
from model_archs.VerticalStackConv2DModelV2 import VerticalStackConv2DModelV2
from model_archs.CustomCNN4_dropout_multi import CustomCNN4_dropout_multi
from model_archs.ResNet18MultiViewAttention import ResNet18MultiViewAttention
from model_archs.SimpleMultiViewNet import SimpleMultiViewNet
from model_archs.ResNetSeparable import ResNetSeparable
from model_archs.SimpleFusionModel import SimpleFusionModel
from model_archs.ResNetSeparableCrossAttention import ResNetSeparableCrossAttention
from model_archs.EnhancedCrossAttentionClass import EnhancedCrossAttentionClass
from model_archs.ResNetSeparableOptimized import ResNetSeparableOptimized
from model_archs.ResNetSeparableOptimized1 import ResNetSeparableOptimized1
from model_archs.ResNetSeparableOptimized2 import ResNetSeparableOptimized2, FocalLoss
from model_archs.ResNet18Separable import ResNet18Separable
from model_archs.ResNet18SeparableAE import ResNet18SeparableAE

import os
import h5py
import numpy as np
import random
import json
import gc
import argparse
import pickle

MODEL_ARCH={
    "SingleStream3D": SingleStream3D,
    "AdvancedConv2DModel": AdvancedConv2DModel,
    "AdvancedConv2DModel1": AdvancedConv2DModel1,
    "AdvancedConv2DModelV2": AdvancedConv2DModelV2,
    "AdvancedConv2DModelV3" : AdvancedConv2DModelV3,
    "VerticalStackConv2DModel": VerticalStackConv2DModel,
    "VerticalStackConv2DModelV1": VerticalStackConv2DModelV1,
    "VerticalStackConv2DModelV2": VerticalStackConv2DModelV2,
    "CustomCNN4_dropout_multi": CustomCNN4_dropout_multi,
    "ResNet18MultiViewAttention" : ResNet18MultiViewAttention,
    "SimpleMultiViewNet" : SimpleMultiViewNet,
    "ResNetSeparable": ResNetSeparable,
    "SimpleFusionModel": SimpleFusionModel,
    "ResNetSeparableCrossAttention": ResNetSeparableCrossAttention,
    "EnhancedCrossAttentionClass" : EnhancedCrossAttentionClass,
    "ResNetSeparableOptimized" : ResNetSeparableOptimized,
    "ResNetSeparableOptimized1" : ResNetSeparableOptimized1,
    "ResNetSeparableOptimized2" : ResNetSeparableOptimized2,
    "ResNet18Separable" : ResNet18Separable,
    "ResNet18SeparableAE" : ResNet18SeparableAE
}

# Custom Modules
from utils import train_model, simple_train_model, save_history, combined_stats, create_data_loaders, create_data_loaders_transform, train_model_Cosine_Annealing, create_test_loaders, extract_model_arch_from_path, plot_corr_matrix, train_model_with_contrastive_loss, train_model_optimized
from utils import val_load_train_image, validate_normalization_via_class, validate_normalization_via_class_no_stat, val_create_data_loaders_no_stat, create_data_loaders_with_scaler
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torchvision import transforms
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
# Data Augmentation transforms (simulated)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])
# transform1 = transforms.Compose([
#     transforms.RandomHorizontalFlip3(),
#     RandomRotate3(),
#     RandomVerticalShift3(),
#     RandomHorizontalShift3(),
# ])


# Load the dataset
# dataset_="livingroom_wholedata"
# dataset_="livingroom_wholedata_sequence_timestep5_stack_vertical"
dataset_="multi_radar"
data_base_dir=f"/home/h3trinh/paper/data/{dataset_}"
folder_train = f"{data_base_dir}/train_data_30"
folder_val = f"{data_base_dir}/val_data_30"
folder_test = f"{data_base_dir}/test_data_30"

    
with open(f'data/wholedata/train_stats.pkl','rb') as f:
    stats = pickle.load(f)

# maps = ['Azimuth','Elevation','Doppler_Map']
# map_stats = dict()
# for m in maps:
#     map_stats[m] = {'mean':0, 'std':1}

# for map_name in maps:
#     means = stats['mean'][map_name]
#     stds = stats['std'][map_name]
#     map_stats[map_name]['mean'],map_stats[map_name]['std'] = combined_stats(means, stds)


# Main function for training
def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_arch",
        type=str,
        required=True,
        dest="model_arch"
    )

    args.add_argument(
        "--is_train",
        # type=bool,
        action="store_true",
        dest="is_train",
    )

    # args.add_argument(
    #     "--is_test",
    #     type=bool,
    #     dest="is_test"
    # )

    args.add_argument(
        "--num_epochs",
        type=int,
        dest="num_epochs"
    )

    args.add_argument(
        "--learning_rate",
        type=float,
        dest="learning_rate"
    )

    args.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size"
    )

    args.add_argument(
        "--optimizer",
        type=str,
        dest="optimizer"
    )

    args.add_argument(
        "--weight_path",
        type=str,
        dest="weight_path"
    )

    parser = args.parse_args()
    model_arch = parser.model_arch
    is_train=parser.is_train
    print(f"is_train: {is_train}")
    # return 1
    # is_test=parser.is_test
    num_epochs=parser.num_epochs
    learning_rate=parser.learning_rate
    batch_size=parser.batch_size
    optimizer=parser.optimizer

    if is_train:
        if model_arch not in MODEL_ARCH.keys():
            # print(f"Not found {model_arch}! Available models arch: {MODEL_ARCH.keys()}")
            raise ValueError(f"Model architecture {args.model_arch} not found.")

    try:
        del model
        del criterion
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass

    
    lr_str = "{:.0e}".format(learning_rate)
    criterion = nn.CrossEntropyLoss()


    if is_train:

        print("TRAINING MODEL...")

        # val_load_train_image(
        #     train_path=folder_train,
        #     map_stats=map_stats
        # 
        # validate_normalization_via_class_no_stat()        

        # model = MODEL_ARCH[model_arch](name=f"_b{batch_size}_lr{lr_str[-1]}_e{num_epochs}_{optimizer}_Cosine_Scheduler", dtype=torch.float32).to(device) 
        model = MODEL_ARCH[model_arch](name=f"_b{batch_size}_lr{lr_str[-1]}_e{num_epochs}_{optimizer}_CosAn_Standard_Scaler_AE", dtype=torch.float32).to(device)
        # print(model)
        # print(list(model.parameters()))
        criterion = nn.CrossEntropyLoss()

        if optimizer == "Adam":
            print(f"Optimizer: {optimizer}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "AdamW":
            print(f"Optimizer: {optimizer}")
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer == "SGD":
            print(f"SGD: {optimizer}")
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # train_loader, val_loader = val_create_data_loaders_no_stat(
        train_loader, val_loader = create_data_loaders_with_scaler(
        # train_loader, val_loader = create_data_loaders_transform(
            folder_train=folder_train,
            folder_val=folder_val,
            # map_stats=map_stats,
            batch_size=batch_size,
            model_arch=model_arch,
            # transform=transform
        )
        # Print the length of val_loader to chek if it's created correctly

        print(f"Length of train_loader {len(train_loader)}")
        print(f"Length of val_loader: {len(val_loader)}")
        save_dir=model.get_dir()

        # Train the model
        # trained_model, history = train_model(
        #     model=model,
        #     train_loader=train_loader,
        #     val_loader=val_loader, 
        #     criterion=criterion, 
        #     optimizer=optimizer, 
        #     epochs=num_epochs, 
        #     scheduler=scheduler,
        #     device=device, 
        #     save_path=save_dir
        # )

        # trained_model, history = train_model_optimized(
        trained_model, history = train_model_Cosine_Annealing(
        # trained_model, history = train_model_with_contrastive_loss(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, 
            epochs=num_epochs,
            lr=learning_rate
        )
        # Save the training history
        save_history(history, save_dir)
    else:
        # Inference

        weight_path = parser.weight_path
        print(f"NOT TRAINING.. LOAD WEIGHT: {weight_path}")

        # print("INFERENCING MODEL...") 
        print(f"NOT TRAINING.. LOAD WEIGHT: {weight_path}")
        model_arch, config = extract_model_arch_from_path(weight_path)

        # print(f"Extracted model architecture: {model_arch}")
        # model = MODEL_ARCH[model_arch](name=f"_b{batch_size}_lr{lr_str[-1]}_e{num_epochs}_{optimizer}_MERGE_resplit", dtype=torch.float32)
        # model = MODEL_ARCH[model_arch](name=f"_b{batch_size}_lr{lr_str[-1]}_e{num_epochs}_{optimizer}_LR", dtype=torch.float32).to(device)
        model = MODEL_ARCH[model_arch](name=f"_b{batch_size}_lr{lr_str[-1]}_e{num_epochs}_{optimizer}_StepLR", dtype=torch.float32).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        
        
        labels_list, preds_list, test_data = create_test_loaders(folder_test, map_stats, batch_size, model_arch, model)
        
        plot_corr_matrix(labels_list=labels_list, preds_list=preds_list, test_data=test_data, model_path=weight_path, model_arch=model_arch, config=config)
    


if __name__ == "__main__":
    main()
