from models import UNetPlusPlus, PADSS, AttenUNetPlusPlus, WNetPlusPlus, UNet, U3UNet
from utils import predict_and_visualize, train_model, train_wnet, train_graph_model, train_unetppgat, invert_border_mask, close_gaps, generate_large_color_space, predict_and_visualize_gattenunetpp, predict_and_visualize_wnet
from loss_functions import DiceFocalLoss, BCEFocalPerceptualLoss, BCETopoLoss, BCEARILoss, CombinedConnectivityLoss, ARIBCEFocalPerceptualLoss
from datasets import AugmentedDataset, AugmentedDatasetv2, OptimizedDataset, TrainingDatasetv2
from acc_unet.ACC_UNet.ACC_UNet import ACC_UNet
from MCNMF_Unet.MCNMFUnet import MCNMF_Unet
from torch_conv_kan.models.u2kanet import u2kagnet_bn_small, u2kagnet_bn
import torchvision.transforms.v2 as v2
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeoDataLoader
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
from scipy.ndimage import label
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
toTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

models = {
    'UNet_vanilla': UNet(3, 4, 3, 0.5, False, 9, False, False, False, device),
    'AttenUNet': UNet(3, 4, 3, 0.5, False, 9, True, False, False, device),
    'UNet_fusion': UNet(3, 4, 3, 0.5, False, 9, False, False, True, device),
    'AttenUNet_fusion': UNet(3, 4, 3, 0.5, False, 9, True, False, True, device),
    'UNetpp_vanilla': UNetPlusPlus(3, 4, 3, False, 9, False, 0.5, device),
    'UNetpp_fusion': UNetPlusPlus(3, 4, 3, False, 9, False, 0.5, True, device),
    'AttenUNetpp': AttenUNetPlusPlus(3, 4, 3, False, 9, False, False, 0.5, device),
    'AttenUNetpp_fusion': AttenUNetPlusPlus(3, 4, 3, False, 9, False, True, 0.5, device),
    'WNetpp_vanilla': WNetPlusPlus(3, 4, 3, False, 9, False, False, False, 0.5, device),
    'WNetpp_fusion': WNetPlusPlus(3, 4, 3, False, 9, False, False, True, 0.5, device),
    'WNetpp_att': WNetPlusPlus(3, 4, 3, False, 9, False, True, False, 0.5, device),
    'WNetpp_att_fusion': WNetPlusPlus(3, 4, 3, False, 9, False, True, True, 0.5, device),
    'U3UNet': U3UNet(3, 2, device)
}

run_directory = 'run_0'
mask_dfs_folder = 'mask_dfs'
model_dfs_folder = 'model_dfs'

dfs = {
    'Normal': {},
    'Drop NANs': {},
    'Dilate and Erode 3x3': {},
    'Dilate and Erode 3x3, drop NANs': {}
    }

truth