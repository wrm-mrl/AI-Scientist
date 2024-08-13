import argparse
from typing import Set, Dict, List
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torchvision.transforms import v2
import os
from PIL import Image, ImageCms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG16_Weights
from torch_geometric.nn import GraphUNet, GraphSAGE, GAT, GIN, GCN
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from skimage.measure import regionprops_table
from scipy.ndimage import label
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
toTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

reference_phi = "../../../samples/FS4_143/cropped_phi/FS4_143_phi_6.png"
reference_phi1 = "../../../samples/FS4_143/cropped_phi1/FS4_143_phi1_6.png"

Rotation_90 = v2.Compose([v2.RandomRotation((90, 90))])
Rotation_180 = v2.Compose([v2.RandomRotation((180, 180))])
Rotation_270 = v2.Compose([v2.RandomRotation((270, 270))])
vflip = v2.Compose([v2.RandomVerticalFlip(1)])
hflip = v2.Compose([v2.RandomHorizontalFlip(1)])
vhflip = v2.Compose([v2.RandomVerticalFlip(1), v2.RandomHorizontalFlip(1)])
augmentations = [Rotation_90, Rotation_180, Rotation_270, vflip, hflip]

def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_graph(image, device='cuda'):
    B, C, H, W = image.shape
    x = image.view(-1, C).to(device)  # (num_nodes, C)
    return x

def create_edges(H, W, device='cuda'):
    edges = []
    for i in range(H):
        for j in range(W):
            node_index = i * W + j
            if i > 0:
                edges.append((node_index, (i - 1) * W + j))
            if i < H - 1:
                edges.append((node_index, (i + 1) * W + j))
            if j > 0:
                edges.append((node_index, i * W + (j - 1)))
            if j < W - 1:
                edges.append((node_index, i * W + (j + 1)))
            if i > 0 and j > 0:
                edges.append((node_index, (i - 1) * W + (j - 1)))
            if i > 0 and j < W - 1:
                edges.append((node_index, (i - 1) * W + (j + 1)))
            if i < H - 1 and j > 0:
                edges.append((node_index, (i + 1) * W + (j - 1)))
            if i < H - 1 and j < W - 1:
                edges.append((node_index, (i + 1) * W + (j + 1)))

    edges = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    return edges

def load_pil_image(image_path, mode='RGB'):
    """
    Load an image from a file path.
    
    Parameters:
    - image_path: Path to the image file.
    
    Returns:
    - image: The loaded image in RGB format.
    """
    return Image.open(image_path).convert(mode)

def rgb_to_lab(image: Image.Image) -> Image.Image:
    """
    Convert an RGB image to the LAB color space.
    
    Parameters:
    - image: Input RGB image.
    
    Returns:
    - lab_image: Image in the LAB color space.
    """
    image = image.convert('RGB')
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    lab_image = ImageCms.profileToProfile(image, srgb_p, lab_p, outputMode='LAB')
    return lab_image

def lab_to_rgb(lab_image: Image.Image) -> Image.Image:
    """
    Convert a LAB image back to the RGB color space.
    
    Parameters:
    - lab_image: Input LAB image.
    
    Returns:
    - rgb_image: Image in the RGB color space.
    """
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb_image = ImageCms.profileToProfile(lab_image, lab_p, srgb_p, outputMode='RGB')
    return rgb_image

def color_normalization(source_image: Image.Image, reference_image: Image.Image) -> Image.Image:
    """
    Perform color normalization from source to reference image.
    
    Parameters:
    - source_image: Source RGB image to be normalized.
    - reference_image: Reference RGB image for normalization.
    
    Returns:
    - normalized_image: Color normalized image in RGB color space.
    """
    # Convert images to LAB color space
    source_lab = rgb_to_lab(source_image)
    reference_lab = rgb_to_lab(reference_image)
    
    # Convert LAB images to numpy arrays
    source_lab_np = np.array(source_lab, dtype=np.float32)
    reference_lab_np = np.array(reference_lab, dtype=np.float32)
    
    # Compute mean and standard deviation of LAB channels
    source_mean, source_std = np.mean(source_lab_np, axis=(0, 1)), np.std(source_lab_np, axis=(0, 1))
    reference_mean, reference_std = np.mean(reference_lab_np, axis=(0, 1)), np.std(reference_lab_np, axis=(0, 1))
    
    # Normalize source image
    normalized_lab_np = (source_lab_np - source_mean) / source_std * reference_std + reference_mean
    normalized_lab_np = np.clip(normalized_lab_np, 0, 255).astype(np.uint8)
    
    # Convert normalized LAB array back to image
    normalized_lab = Image.fromarray(normalized_lab_np, mode='LAB')
    
    # Convert LAB image back to RGB color space
    normalized_image = lab_to_rgb(normalized_lab)
    return normalized_image

def close_gaps(mask, dialation_kernel_size, erosion_kernel_size):
    # Define the kernel for morphological operations
    kernel = torch.ones((1, 1, dialation_kernel_size, dialation_kernel_size), device=mask.device, dtype=mask.dtype)
    kernel_2 = torch.ones((1, 1, erosion_kernel_size, erosion_kernel_size), device=mask.device, dtype=mask.dtype)

    # Apply dilation followed by erosion (closing)
    dilation = F.conv2d(mask, kernel, padding=0)
    dilation = (dilation > 0).float()
    closed_mask = F.conv2d(dilation, kernel_2, padding=0)

    closed_mask = (closed_mask == kernel_2.sum()).float()
    return closed_mask

def invert_border_mask(mask):
    return (mask == 0.).float()

class TrainingDatasetv2(Dataset):
    def __init__(self, top_folder='../../training_data_512', amp_folder='amp', Phi_folder='Phi', phi1_folder='phi1', mask_folder='mask', for_graph_model: bool=False, device=device):
        self.device = device
        self.images = []
        self.masks = []
        self.edges = create_edges(512, 512) if for_graph_model else None
        reference_Phi_image = load_pil_image(reference_phi)
        reference_phi1_image = load_pil_image(reference_phi1)
        
        amp_im_paths = sorted(os.listdir(f'{top_folder}/{amp_folder}/'))
        Phi_im_paths = sorted(os.listdir(f'{top_folder}/{Phi_folder}/'))
        phi1_im_paths = sorted(os.listdir(f'{top_folder}/{phi1_folder}/'))
        mask_im_paths = sorted(os.listdir(f'{top_folder}/{mask_folder}/'))
        assert len(amp_im_paths) == len(Phi_im_paths) == len(phi1_im_paths) == len(mask_im_paths)
        for i in range(len(amp_im_paths)):
            normalized_Phi = color_normalization(load_pil_image(f'{top_folder}/{Phi_folder}/{Phi_im_paths[i]}', 'L'), reference_Phi_image).convert('L')
            normalized_phi1 = color_normalization(load_pil_image(f'{top_folder}/{phi1_folder}/{phi1_im_paths[i]}', 'L'), reference_phi1_image).convert('L')
            amp = load_pil_image(f'{top_folder}/{amp_folder}/{amp_im_paths[i]}', 'I;16')
            im = torch.cat([toTensor(amp), toTensor(normalized_Phi), toTensor(normalized_phi1)], dim=0)
            mask = toTensor(load_pil_image(f'{top_folder}/{mask_folder}/{mask_im_paths[i]}', '1'))
            if for_graph_model:
                self.images.append(Data(x=create_graph(im.unsqueeze(0), 'cpu'), edge_index=self.edges))
            else:
                self.images.append(im)
            self.masks.append(mask)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx].to(self.device), self.masks[idx].to(self.device)

def get_data(batch_size):
    train_dataset = TrainingDatasetv2('../../../training_data_512')
    val_dataset = TrainingDatasetv2('../../../testing/testing_data_512')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

''' Attention Block Model Class '''
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

''' Position Attention Block Model Class '''
class PositionAttentionBlock(nn.Module):
    def __init__(self, in_channels, device=torch.device('cuda')):
        super(PositionAttentionBlock, self).__init__()
        self.device = device
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), 1, device=device)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), 1, device=device)
        self.g = nn.Conv2d(int(in_channels / 2), int(in_channels / 2), 1, device=device)
        self.softmax = nn.Softmax(1)
        
        def CBR(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        self.cbr = CBR(int(in_channels / 2), int(in_channels / 2))
    
    def forward(self, H, L):
        HL = torch.cat((H, L), dim=1).to(self.device)
        A = self.theta(HL)
        B = self.phi(HL)
        A = torch.reshape(A, (A.shape[0], A.shape[1], A.shape[2] * A.shape[3]))
        B = torch.reshape(B, (B.shape[0], B.shape[1], B.shape[2] * B.shape[3]))
        A = torch.transpose(A, 1, 2)
        M = torch.bmm(A, B)
        M = self.softmax(M)
        D = torch.reshape(self.g(L), (L.shape[0], int(L.shape[1]), L.shape[2] * L.shape[3]))
        E = torch.bmm(D, M)
        E = torch.reshape(E, (L.shape[0], int(L.shape[1]), L.shape[2], L.shape[3]))
        F = self.cbr(E)
        G = torch.add(H, F)
        return G

''' Channel Attention Blcok Model Class '''
class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, gp_in_channels: dict, device='cuda'):
        super(ChannelAttentionBlock, self).__init__()
        self.device=device
        self.gp = nn.AdaptiveAvgPool2d(gp_in_channels[str(in_channels)])
        self.c1 = nn.Conv2d(in_channels, in_channels, 1, device=device)
        self.r1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(in_channels, int(in_channels / 2), 1, device=device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, H, L):
        HL = torch.cat((H, L), dim=1).to(self.device)
        A = self.gp(HL)
        B = self.c1(A)
        C = self.r1(B)
        D = self.c2(C)
        E = self.sigmoid(D)
        F = torch.zeros_like(H, device=self.device)
        G = torch.zeros_like(H, device=self.device)
        for i in range(F.shape[0]):
            F[i] = torch.mul(H[i], E[i])
            G[i] = torch.add(F[i], L[i])
        return G

''' Neighborhood Similarity Layer Model Class '''
class NeighborhoodSimilarityLayer(nn.Module):
    def __init__(self, in_channels, neighborhood_size=3, device='cuda'):
        super(NeighborhoodSimilarityLayer, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.in_channels = in_channels
        self.device = device

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == self.in_channels
        similarities = torch.zeros(B, self.neighborhood_size ** 2 - 1, H, W).to(dtype=x.dtype, device=self.device)
        for i in range(B):
            z = x[i].unsqueeze(0)
            # Compute the mean feature vector
            mean_feature_vector = torch.mean(z, dim=[2, 3], keepdim=True)
            mean_feature_vector = mean_feature_vector.permute(0, 2, 3, 1).unsqueeze(-1)

            # Pad the input tensor
            pad_size = self.neighborhood_size // 2
            x_padded = F.pad(z, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

            patches = x_padded.unfold(2, self.neighborhood_size, 1).unfold(3, self.neighborhood_size, 1)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(1, H, W, C, -1)

            # Exclude the center pixel
            center_index = (self.neighborhood_size ** 2) // 2
            patches_excluding_center = torch.cat((patches[..., :center_index], patches[..., center_index+1:]), dim=-1)

            x_center = z.permute(0, 2, 3, 1).unsqueeze(-1)
            x_center = x_center.expand_as(patches_excluding_center)

            # Center the feature vectors
            centered_patches = patches_excluding_center - mean_feature_vector
            centered_x_center = x_center - mean_feature_vector

            # Calculate the normalized inner product
            numerator = torch.sum(centered_patches * centered_x_center, dim=3)
            denominator = torch.sqrt(torch.sum(centered_patches ** 2, dim=3) * torch.sum(centered_x_center ** 2, dim=3))

            # Prevent division by zero
            denominator = torch.clamp(denominator, min=1e-8)
            similarity = numerator / denominator

            similarity = similarity.permute(0, 3, 1, 2).contiguous()
            similarities[i] = similarity
        return similarities
    
''' UNet Model Class '''
class UNet(nn.Module):
    def __init__(self, in_channels: int=3, scale_factor: int=4, kernel_size: int=3, dropout=0, use_nsl: bool=False, neighborhood_size: int=9, attention: bool=False, deep: bool=False, fusion_module: bool=False, device=device):
        super(UNet, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            halfway = int((in_channels + out_channels) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, halfway, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(halfway, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(halfway, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
            )
        
        def CBRD(in_channels, out_channels, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            halfway = int((in_channels + out_channels) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, halfway, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(halfway, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(halfway, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        
        def CBRNSL(in_channels: int, out_channels: int, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(in_channels, device=device),
                nn.ReLU(inplace=True),
                NeighborhoodSimilarityLayer(in_channels, neighborhood_size, device),
                nn.Conv2d(neighborhood_size ** 2 - 1, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        
        self.deep = deep
        if use_nsl:
            self.enc1 = CBRNSL(in_channels, 16 * scale_factor)
        else:
            self.enc1 = CBRD(in_channels, 16 * scale_factor)
        self.enc2 = CBRD(16 * scale_factor, 32 * scale_factor)
        self.enc3 = CBRD(32 * scale_factor, 64 * scale_factor)
        self.enc4 = CBRD(64 * scale_factor, 128 * scale_factor)
        self.center = CBRD(128 * scale_factor, 256 * scale_factor)
        
        self.dec4 = CBR(256 * scale_factor, 128 * scale_factor)
        self.dec3 = CBR(128 * scale_factor, 64 * scale_factor)
        self.dec2 = CBR(64 * scale_factor, 32 * scale_factor)
        self.dec1 = CBR(32 * scale_factor, 16 * scale_factor)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.ConvTranspose2d(256 * scale_factor, 128 * scale_factor, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(128 * scale_factor, 64 * scale_factor, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(64 * scale_factor, 32 * scale_factor, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(32 * scale_factor, 16 * scale_factor, kernel_size=2, stride=2)

        self.attention = attention
        if attention:
            self.attention4 = AttentionBlock(F_g=128 * scale_factor, F_l=128 * scale_factor, F_int=64 * scale_factor)
            self.attention3 = AttentionBlock(F_g=64 * scale_factor, F_l=64 * scale_factor, F_int=32 * scale_factor)
            self.attention2 = AttentionBlock(F_g=32 * scale_factor, F_l=32 * scale_factor, F_int=16 * scale_factor)
            self.attention1 = AttentionBlock(F_g=16 * scale_factor, F_l=16 * scale_factor, F_int=8 * scale_factor)

        self.using_late_fusion_module = fusion_module
        if fusion_module:
            self.fusion_upsample1 = nn.ConvTranspose2d(64 * scale_factor, 32 * scale_factor, kernel_size=2, stride=2)
            self.fusion_upsample2 = nn.ConvTranspose2d(32 * scale_factor, 16 * scale_factor, kernel_size=2, stride=2)
            self.fusion_cbr = CBR(48 * scale_factor, 16 * scale_factor)
        
        self.out = nn.Conv2d(16 * scale_factor, 1, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        
        center = self.center(self.maxpool(enc4))

        dec4 = self.upsample(center)
        if self.attention:
            enc4 = self.attention4(g=dec4, x=enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upsample4(dec4)
        if self.attention:
            enc3 = self.attention3(g=dec3, x=enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upsample3(dec3)
        if self.attention:
            enc2 = self.attention2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upsample2(dec2)
        if self.attention:
            enc1 = self.attention1(g=dec1, x=enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        if self.using_late_fusion_module:
            dec2 = self.fusion_upsample2(dec2)
            dec3 = self.fusion_upsample1(dec3)
            dec3 = self.fusion_upsample2(dec3)

            ## fusion module
            out = self.fusion_cbr(torch.cat([dec1, dec2, dec3], dim=1))
            out = self.out(out)
        else:
            out = self.out(dec1)
        
        if self.deep:
            return [out, dec1, dec2, dec3, dec4]
        return out

''' UNet++ Model Class '''
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int=3, scale_factor: int = 4, kernel_size: int=3, use_nsl: bool=True, neighborhood_size: int=3, deep: bool=True, dropout=0.15, late_fusion: bool=False, device='cuda'):
        super(UNetPlusPlus, self).__init__()

        assert kernel_size in [1, 3, 5, 7], "kernel_size must be 1, 3, 5, or 7"

        def CBR(in_channels, out_channels, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            halfway = int((in_channels + out_channels) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, halfway, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(halfway, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(halfway, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
            )
        
        def CBRD(in_channels, out_channels, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            halfway = int((in_channels + out_channels) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, halfway, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(halfway, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(halfway, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        
        def CBRNSL(in_channels: int, out_channels: int, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            mid_channels = max(in_channels, 3)
            return nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(mid_channels, device=device),
                nn.ReLU(inplace=True),
                NeighborhoodSimilarityLayer(mid_channels, neighborhood_size, device),
                nn.Conv2d(neighborhood_size ** 2 - 1, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        
        self.deep = deep
        if use_nsl:
            self.enc1 = CBRNSL(in_channels, 16 * scale_factor)
        else:
            self.enc1 = CBRD(in_channels, 16 * scale_factor)
        self.enc2 = CBRD(16 * scale_factor, 32 * scale_factor)
        self.enc3 = CBRD(32 * scale_factor, 64 * scale_factor)
        self.enc4 = CBRD(64 * scale_factor, 128 * scale_factor)
        self.center = CBRD(128 * scale_factor, 256 * scale_factor)

        self.x_01 = CBR(32 * scale_factor, 16 * scale_factor)
        self.x_02 = CBR(48 * scale_factor, 16 * scale_factor)
        self.x_03 = CBR(64 * scale_factor, 16 * scale_factor)
        self.x_04 = CBR(80 * scale_factor, 16 * scale_factor)

        self.x_11 = CBR(64 * scale_factor, 32 * scale_factor)
        self.x_21 = CBR(128 * scale_factor, 64 * scale_factor)
        self.x_31 = CBR(256 * scale_factor, 128 * scale_factor)
        self.x_12 = CBR(96 * scale_factor, 32 * scale_factor)
        self.x_22 = CBR(192 * scale_factor, 64 * scale_factor)
        self.x_13 = CBR(128 * scale_factor, 32 * scale_factor)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample512 = nn.ConvTranspose2d(256 * scale_factor, 128 * scale_factor, kernel_size=2, stride=2, device=device)
        self.upsample256 = nn.ConvTranspose2d(128 * scale_factor, 64 * scale_factor, kernel_size=2, stride=2, device=device)
        self.upsample128 = nn.ConvTranspose2d(64 * scale_factor, 32 * scale_factor, kernel_size=2, stride=2, device=device)
        self.upsample64 = nn.ConvTranspose2d(32 * scale_factor, 16 * scale_factor, kernel_size=2, stride=2, device=device)

        self.late_fusion = late_fusion
        if late_fusion:
            self.fusion_out = CBR(48 * scale_factor, 16 * scale_factor, kernel_size=3)

        self.out = nn.Conv2d(16 * scale_factor, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        center = self.center(self.maxpool(enc4))

        x_01 = self.x_01(torch.cat([enc1, self.upsample64(enc2)], dim=1))
        x_11 = self.x_11(torch.cat([enc2, self.upsample128(enc3)], dim=1))
        x_21 = self.x_21(torch.cat([enc3, self.upsample256(enc4)], dim=1))
        x_31 = self.x_31(torch.cat([enc4, self.upsample512(center)], dim=1))

        x_02 = self.x_02(torch.cat([enc1, x_01, self.upsample64(x_11)], dim=1))
        x_12 = self.x_12(torch.cat([enc2, x_11, self.upsample128(x_21)], dim=1))
        x_22 = self.x_22(torch.cat([enc3, x_21, self.upsample256(x_31)], dim=1))

        x_03 = self.x_03(torch.cat([enc1, x_01, x_02, self.upsample64(x_12)], dim=1))
        x_13 = self.x_13(torch.cat([enc2, x_11, x_12, self.upsample128(x_22)], dim=1))

        x_04 = self.x_04(torch.cat([enc1, x_01, x_02, x_03, self.upsample64(x_13)], dim=1))

        if self.late_fusion:
            x_04 = self.fusion_out(torch.cat([x_04, x_03, x_02], dim=1))

        if self.deep:
            return [self.out(x_04), self.out(x_03), self.out(x_02), self.out(x_01)]
        else:
            return self.out(x_04)

''' AttenUnet++ Model Class '''
class AttenUNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int=3, scale_factor: int = 4, kernel_size: int=3, use_nsl: bool=True, neighborhood_size: int=3, deep: bool=True, late_fusion: bool=False, dropout=0.15, device='cuda'):
        super(AttenUNetPlusPlus, self).__init__()

        assert kernel_size in [1, 3, 5, 7], "kernel_size must be 1, 3, 5, or 7"

        def CBR(in_channels, out_channels, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            halfway = int((in_channels + out_channels) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, halfway, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(halfway, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(halfway, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
            )
        
        def CBRD(in_channels, out_channels, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            halfway = int((in_channels + out_channels) / 2)
            return nn.Sequential(
                nn.Conv2d(in_channels, halfway, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(halfway, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(halfway, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        
        def CBRNSL(in_channels: int, out_channels: int, kernel_size: int=kernel_size):
            padding = int((kernel_size - 1) / 2)
            mid_channels = max(in_channels, 3)
            return nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(mid_channels, device=device),
                nn.ReLU(inplace=True),
                NeighborhoodSimilarityLayer(mid_channels, neighborhood_size, device),
                nn.Conv2d(neighborhood_size ** 2 - 1, out_channels, kernel_size=kernel_size, padding=padding, device=device),
                nn.BatchNorm2d(out_channels, device=device),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        
        self.deep = deep
        if use_nsl:
            self.enc1 = CBRNSL(in_channels, 16 * scale_factor)
        else:
            self.enc1 = CBRD(in_channels, 16 * scale_factor)
        self.enc2 = CBRD(16 * scale_factor, 32 * scale_factor)
        self.enc3 = CBRD(32 * scale_factor, 64 * scale_factor)
        self.enc4 = CBRD(64 * scale_factor, 128 * scale_factor)
        self.center = CBRD(128 * scale_factor, 256 * scale_factor)

        self.attention4 = AttentionBlock(F_g=128 * scale_factor, F_l=128 * scale_factor, F_int=64 * scale_factor)
        self.attention3 = AttentionBlock(F_g=64 * scale_factor, F_l=64 * scale_factor, F_int=32 * scale_factor)
        self.attention2 = AttentionBlock(F_g=32 * scale_factor, F_l=32 * scale_factor, F_int=16 * scale_factor)
        self.attention1 = AttentionBlock(F_g=16 * scale_factor, F_l=16 * scale_factor, F_int=8 * scale_factor)

        self.x_01 = CBR(32 * scale_factor, 16 * scale_factor)
        self.x_02 = CBR(48 * scale_factor, 16 * scale_factor)
        self.x_03 = CBR(64 * scale_factor, 16 * scale_factor)
        self.x_04 = CBR(80 * scale_factor, 16 * scale_factor)

        self.x_11 = CBR(64 * scale_factor, 32 * scale_factor)
        self.x_21 = CBR(128 * scale_factor, 64 * scale_factor)
        self.x_31 = CBR(256 * scale_factor, 128 * scale_factor)
        self.x_12 = CBR(96 * scale_factor, 32 * scale_factor)
        self.x_22 = CBR(192 * scale_factor, 64 * scale_factor)
        self.x_13 = CBR(128 * scale_factor, 32 * scale_factor)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample512 = nn.ConvTranspose2d(256 * scale_factor, 128 * scale_factor, kernel_size=2, stride=2, device=device)
        self.upsample256 = nn.ConvTranspose2d(128 * scale_factor, 64 * scale_factor, kernel_size=2, stride=2, device=device)
        self.upsample128 = nn.ConvTranspose2d(64 * scale_factor, 32 * scale_factor, kernel_size=2, stride=2, device=device)
        self.upsample64 = nn.ConvTranspose2d(32 * scale_factor, 16 * scale_factor, kernel_size=2, stride=2, device=device)

        self.late_fusion = late_fusion
        if late_fusion:
            self.fusion_out = CBR(48 * scale_factor, 16 * scale_factor, kernel_size=3)

        self.out = nn.Conv2d(16 * scale_factor, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        center = self.center(self.maxpool(enc4))

        upsampled_center = self.upsample512(center)
        enc4_attention = self.attention4(upsampled_center, enc4)
        x_31 = self.x_31(torch.cat([enc4_attention, upsampled_center], dim=1))

        upsampled_x_31 = self.upsample256(x_31)
        enc3_attention = self.attention3(upsampled_x_31, enc3)
        x_21 = self.x_21(torch.cat([enc3_attention, upsampled_x_31], dim=1))

        upsampled_x_21 = self.upsample128(x_21)
        enc2_attention = self.attention2(upsampled_x_21, enc2)
        x_11 = self.x_11(torch.cat([enc2_attention, upsampled_x_21], dim=1))

        upsampled_x_11 = self.upsample64(x_11)
        enc1_attention = self.attention1(upsampled_x_11, enc1)
        x_01 = self.x_01(torch.cat([enc1_attention, upsampled_x_11], dim=1))

        x_02 = self.x_02(torch.cat([enc1_attention, x_01, upsampled_x_11], dim=1))
        x_12 = self.x_12(torch.cat([enc2_attention, x_11, upsampled_x_21], dim=1))
        x_22 = self.x_22(torch.cat([enc3_attention, x_21, upsampled_x_31], dim=1))

        x_03 = self.x_03(torch.cat([enc1_attention, x_01, x_02, self.upsample64(x_12)], dim=1))
        x_13 = self.x_13(torch.cat([enc2_attention, x_11, x_12, self.upsample128(x_22)], dim=1))

        x_04 = self.x_04(torch.cat([enc1_attention, x_01, x_02, x_03, self.upsample64(x_13)], dim=1))

        if self.late_fusion:
            x_04 = self.fusion_out(torch.cat([x_04, x_03, x_02], dim=1))

        if self.deep:
            return [self.out(x_04), self.out(x_03), self.out(x_02), self.out(x_01)]
        else:
            return self.out(x_04)

''' WNet Model Class '''
class WNet(nn.Module):
    def __init__(self, in_channels: int=3, scale_factor: int = 4, kernel_size: int=3, use_nsl: bool=True, neighborhood_size: int=3, deep: bool=True, attention: bool=True, late_fusion: bool=False, dropout=0.15, device='cuda'):
        super(WNet, self).__init__()

        self.u1 = UNet(in_channels, scale_factor, kernel_size, dropout, False, neighborhood_size, attention, False, late_fusion, device)
        self.u2 = UNet(1, 1, kernel_size, 0, use_nsl, neighborhood_size, attention, False, late_fusion, device)
        self.deep = deep
    
    def forward(self, x):
        u1 = self.u1(x)
        y = self.u2(u1)
        if self.deep:
            return [y, u1]
        return y

''' WNet++ Model Class '''
class WNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int=3, scale_factor: int = 4, kernel_size: int=3, use_nsl: bool=True, neighborhood_size: int=3, deep: bool=True, attention: bool=True, late_fusion: bool=False, dropout=0.15, device='cuda'):
        super(WNetPlusPlus, self).__init__()
        if attention:
            self.u1 = AttenUNetPlusPlus(in_channels, scale_factor, kernel_size, False, neighborhood_size, False, late_fusion, dropout, device)
            self.u2 = AttenUNetPlusPlus(1, 1, kernel_size, use_nsl, neighborhood_size, False, late_fusion, 0, device)
        else:
            self.u1 = UNetPlusPlus(in_channels, scale_factor, kernel_size, False, neighborhood_size, False, dropout, late_fusion, device)
            self.u2 = UNetPlusPlus(1, 1, kernel_size, use_nsl, neighborhood_size, False, 0, late_fusion, device)
        self.deep = deep
    
    def forward(self, x):
        u1 = self.u1(x)
        y = self.u2(u1)
        if self.deep:
            return [y, u1]
        return y

''' UNet++Graph Model Class '''
class GraphUNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int=3, scale_factor: int = 4, kernel_size: int=3, use_nsl: bool=True, neighborhood_size: int=3, deep: bool=False, hidden_size: int=32, num_layers: int=2, graph_model: str='gat', device='cuda'):
        super(GraphUNetPlusPlus, self).__init__()

        self.device = device
        self.deep = deep
        self.unet_only = False
        self.u = UNetPlusPlus(in_channels, scale_factor, kernel_size, use_nsl, neighborhood_size, deep, 0.2, device)
        if deep:
            self.g = GAT(4, hidden_size, num_layers, 1, v2=True)
        else:
            if graph_model == 'gat':
                self.g = GAT(1, hidden_size, num_layers, 1, v2=True)
            elif graph_model == 'unet':
                self.g = GraphUNet(1, hidden_size, 1, num_layers)
            elif graph_model == 'sage':
                self.g = GraphSAGE(1, hidden_size, num_layers, 1)
            elif graph_model == 'gin':
                self.g = GIN(1, hidden_size, num_layers, 1)
            elif graph_model == 'gcn':
                self.g = GCN(1, hidden_size, num_layers, 1)
            else:
                assert graph_model in ['gat', 'unet', 'sage', 'gin', 'gcn'], 'Unknown graph model specified'
            #self.gat = GAT(in_channels=1, out_channels=1, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=0, mask_size=1, device=device)
    
    def create_edges(self, H, W, device=device):
        edges = []
        for i in range(H):
            for j in range(W):
                node_index = i * W + j
                if i > 0:
                    edges.append((node_index, (i - 1) * W + j))
                if i < H - 1:
                    edges.append((node_index, (i + 1) * W + j))
                if j > 0:
                    edges.append((node_index, i * W + (j - 1)))
                if j < W - 1:
                    edges.append((node_index, i * W + (j + 1)))
                if i > 0 and j > 0:
                    edges.append((node_index, (i - 1) * W + (j - 1)))
                if i > 0 and j < W - 1:
                    edges.append((node_index, (i - 1) * W + (j + 1)))
                if i < H - 1 and j > 0:
                    edges.append((node_index, (i + 1) * W + (j - 1)))
                if i < H - 1 and j < W - 1:
                    edges.append((node_index, (i + 1) * W + (j + 1)))

        edges = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        return edges

    def forward(self, x):
        unetpp_output = self.u(x)
        if self.unet_only:
            return unetpp_output
        if self.deep:
            gat_input = torch.sigmoid(torch.cat(unetpp_output, dim=1).view(-1, 4))
        else:
            gat_input = unetpp_output.view(-1, 1)
            #gat_input = torch.sigmoid(unetpp_output.view(-1, 1))
        gat_output = self.g(gat_input, self.create_edges(x.shape[2], x.shape[3], self.device))
        if self.deep:
            return unetpp_output[0], gat_output
        else:
            return gat_output

''' U3UNet++ Model Class '''
class U3UNet(nn.Module):
    def __init__(self, in_chans=3, scale_factor=2, device=device):
        super(U3UNet, self).__init__()
        self.u1 = UNetPlusPlus(in_chans, scale_factor, 3, False, 9, False, 0.5, False, device)
        self.u2 = UNetPlusPlus(in_chans, scale_factor, 5, False, 9, False, 0.5, False, device)
        self.u3 = UNetPlusPlus(in_chans, scale_factor, 7, False, 9, False, 0.5, False, device)
        self.w = UNetPlusPlus(3, 1, 3, False, 9, False, 0, False, device)

    def forward(self, x):
        u1 = self.u1(x)
        u2 = self.u2(x)
        u3 = self.u3(x)
        w = self.w(torch.cat([u1, u2, u3], dim=1))
        return w

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce_exp = torch.exp(-bce)
        focal = self.alpha * (1 - bce_exp) ** self.gamma * bce
        return focal.mean()

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.use_gpu = use_gpu
        self.vgg = self._get_vgg_features(layers)
        self.layer_mapping = {
            'conv1_2': '4',
            'conv2_2': '9',
            'conv3_3': '16',
            'conv4_3': '23',
            'conv5_3': '30',
        }
        self.layer_names = layers
        if self.use_gpu:
            self.vgg = self.vgg.cuda()

    def _get_vgg_features(self, layers):
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        model = nn.Sequential()
        layer_mapping = {
            'conv1_2': 4,
            'conv2_2': 9,
            'conv3_3': 16,
            'conv4_3': 23,
            'conv5_3': 30,
        }
        for name, module in vgg._modules.items():
            model.add_module(name, module)
            if name in [str(layer_mapping[layer]) for layer in layers]:
                model.add_module("relu_{}".format(name), nn.ReLU())
        return model

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        inputs_features = self._extract_features(inputs)
        targets_features = self._extract_features(targets)
        loss = 0
        for name in self.layer_names:
            layer_index = self.layer_mapping[name]
            loss += nn.functional.mse_loss(inputs_features[layer_index], targets_features[layer_index])
        return loss

    def _extract_features(self, x):
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_mapping.values():
                features[name] = x
        return features

class BCEFocalPerceptualLoss(nn.Module):
    def __init__(self, focal_alpha=1, focal_gamma=2, focal_smooth=1e-5, bce_weight=0.5, focal_weight=1, perceptual_weight=0.1):
        super(BCEFocalPerceptualLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(focal_alpha, focal_gamma, focal_smooth)
        self.perceptual = PerceptualLoss()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, inputs, targets):
        return self.bce_weight * self.bce(inputs, targets) + self.focal_weight * self.focal(inputs, targets) + self.perceptual_weight * self.perceptual(inputs, targets)

def train_model(model, data_loader: DataLoader, criterion, optimizer, num_epochs: int=1, device='cuda') -> dict:
    scaler = GradScaler()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with autocast(device, cache_enabled=False):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            del images, masks
            torch.cuda.empty_cache()
    return {'train_loss': epoch_loss}

def evaluate_model(model, val_loader: DataLoader, criterion, num_epochs: int=1, device='cuda') -> dict:
    for epoch in range(num_epochs):
        model.eval()
        epoch_loss = 0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                pred = model(images)
            loss = criterion(pred, masks)
            epoch_loss += loss.item()
    return {'val_loss': epoch_loss}

def run(out_dir, dataset, seed_offset, num_epochs=400):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    train_loader, val_loader = get_data(batch_size=8)

    models = {
        'UNet_vanilla': UNet(3, 4, 3, 0.5, False, 9, False, False, False, device),
        'AttenUNet': UNet(3, 4, 3, 0.5, False, 9, True, False, False, device),
        'UNet_fusion': UNet(3, 4, 3, 0.5, False, 9, False, False, True, device),
        'AttenUNet_fusion': UNet(3, 4, 3, 0.5, False, 9, True, False, True, device),
        #'UNet_nsl': UNet(3, 3, 0.3, True, 9, False, False, False, device),
        #'UNet_nsl_fusion': UNet(3, 3, 0.3, True, 9, False, False, True, device),
        #'AttenUNet_nsl_fusion': UNet(3, 3, 0.3, True, 9, True, False, True, device),
        'UNetpp_vanilla': UNetPlusPlus(3, 4, 3, False, 9, False, 0.5, device),
        'UNetpp_fusion': UNetPlusPlus(3, 4, 3, False, 9, False, 0.5, True, device),
        'AttenUNetpp': AttenUNetPlusPlus(3, 4, 3, False, 9, False, False, 0.5, device),
        'AttenUNetpp_fusion': AttenUNetPlusPlus(3, 4, 3, False, 9, False, True, 0.5, device),
        #'AttenUNetpp_nsl': AttenUNetPlusPlus(3, 4, 3, True, 9, False, False, 0.3, device),
        'WNetpp_vanilla': WNetPlusPlus(3, 4, 3, False, 9, False, False, False, 0.5, device),
        'WNetpp_fusion': WNetPlusPlus(3, 4, 3, False, 9, False, False, True, 0.5, device),
        #'WNetpp_nsl': WNetPlusPlus(3, 4, 3, True, 9, False, False, False, 0.3, device),
        'WNetpp_att': WNetPlusPlus(3, 4, 3, False, 9, False, True, False, 0.5, device),
        #'WNetpp_nsl_att': WNetPlusPlus(3, 4, 3, True, 9, False, True, False, 0.3, device),
        'WNetpp_att_fusion': WNetPlusPlus(3, 4, 3, False, 9, False, True, True, 0.5, device),
        'U3UNet': U3UNet(3, 2, device)
        #'PADSS': PADSS(3, 3, 16, False, 0.2, device),
        #'ACC_UNet': ACC_UNet(3, 1, 16),
        #'u2kagnet_bn': u2kagnet_bn(3, 1, 1, 3, 4, 0.75),
        #'UNetppSAGE': GraphUNetPlusPlus(3, 4, 3, False, 9, False, 8, 3, 'sage', device),
        #'UNetppGCN': GraphUNetPlusPlus(3, 4, 3, False, 9, False, 8, 3, 'gcn', device),
        #'UNetppGAT': GraphUNetPlusPlus(3, 4, 3, False, 9, False, 8, 3, 'gat', device),
        #'U2KagNetSAGE': U2KagNetGraph(3, 4, 8, 2, 0.75, 'sage', False, device),
    }
    if 'mask_dfs' not in os.listdir(out_dir):
        os.makedirs(os.path.join(out_dir, 'mask_dfs'), exist_ok=True)
        master_mask_df = pd.DataFrame()
        master_mask_drop_df = pd.DataFrame()
        master_m33_df = pd.DataFrame()
        master_m33_drop_df = pd.DataFrame()
        for _, masks in val_loader:
            for i in range(masks.shape[0]):
                mask = masks[i].unsqueeze(0)
                m33 = close_gaps(mask, 3, 3)
                mask_inv = invert_border_mask(mask)
                m33_inv = invert_border_mask(m33)
                labeled_mask, _ = label(mask_inv[0][0].detach().cpu().numpy())
                labeled_m33, _ = label(m33_inv[0][0].detach().cpu().numpy())
                properties = ['label', 'area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'centroid', 'orientation',
                              'eccentricity', 'moments_hu']
                regions_table = regionprops_table(labeled_mask, properties=properties)
                regions_table33 = regionprops_table(labeled_m33, properties=properties)
                mask_df = pd.DataFrame(regions_table)
                m33_df = pd.DataFrame(regions_table33)
                mask_df['aspect_ratio'] = mask_df['major_axis_length'] / mask_df['minor_axis_length']
                mask_df['equivalent_diameter'] = np.sqrt(4 * mask_df['area'] / np.pi)
                m33_df['aspect_ratio'] = m33_df['major_axis_length'] / m33_df['minor_axis_length']
                m33_df['equivalent_diameter'] = np.sqrt(4 * m33_df['area'] / np.pi)
                mask_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                m33_df.replace([np.inf, -np.inf], np.nan, inplace=True)

                mask_df_drop = mask_df.copy()
                m33_df_drop = m33_df.copy()

                mask_df_drop.dropna(inplace=True)
                m33_df_drop.dropna(inplace=True)

                master_mask_df = pd.concat([master_mask_df, mask_df], ignore_index=True)
                master_mask_drop_df = pd.concat([master_mask_drop_df, mask_df_drop], ignore_index=True)
                master_m33_df = pd.concat([master_m33_df, m33_df], ignore_index=True)
                master_m33_drop_df = pd.concat([master_m33_drop_df, m33_df_drop], ignore_index=True)

        master_mask_df.to_csv(f'{out_dir}/mask_dfs/mask_nodrop.csv')
        master_mask_drop_df.to_csv(f'{out_dir}/mask_dfs/mask_drop.csv')
        master_m33_df.to_csv(f'{out_dir}/mask_dfs/mask_d3_e3_nodrop.csv')
        master_m33_drop_df.to_csv(f'{out_dir}/mask_dfs/mask_d3_e3_drop.csv')

    master_info = {}
    master_train_log = {}
    master_val_log = {}
    for model_name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.98), weight_decay=0.1)
       
        warmup_steps = 50
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: min(s / warmup_steps, 1)
        )

        pbar = tqdm(range(num_epochs), f'Training {model_name}')
        final_info, train_log_info, val_log_info = [], [], []
        for ep in pbar:
            train_metrics = train_model(model, train_loader, BCEFocalPerceptualLoss(), optimizer, 1)
            val_metrics = evaluate_model(model, val_loader, BCEFocalPerceptualLoss())
            scheduler.step()
            train_metrics["step"] = (ep + 1)
            val_metrics["step"] = (ep + 1)

            # if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
            #     step_val_acc_99 = val_metrics["step"]
            train_log_info.append(train_metrics)
            val_log_info.append(val_metrics)


        final_info = {
            "final_train_loss": train_metrics["train_loss"],
            "final_val_loss": val_metrics["val_loss"],
            #"final_train_acc": train_metrics["train_accuracy"],
            #"final_val_acc": val_metrics["val_accuracy"],
            #"step_val_acc_99": step_val_acc_99,
        }
        master_info[model_name] = final_info
        master_train_log[model_name] = train_log_info
        master_val_log[model_name] = val_log_info
        print(final_info)

        master_mask_df = pd.DataFrame()
        master_mask_drop_df = pd.DataFrame()
        master_m33_df = pd.DataFrame()
        master_m33_drop_df = pd.DataFrame()

        for images, _ in val_loader:
            for i in range(images.shape[0]):
                model.eval()
                mask = model(images[i].unsqueeze(0))
                m33 = close_gaps(mask, 3, 3)
                mask_inv = invert_border_mask(mask)
                m33_inv = invert_border_mask(m33)
                labeled_mask, _ = label(mask_inv[0][0].detach().cpu().numpy())
                labeled_m33, _ = label(m33_inv[0][0].detach().cpu().numpy())
                properties = ['label', 'area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'centroid', 'orientation',
                              'eccentricity', 'moments_hu']
                regions_table = regionprops_table(labeled_mask, properties=properties)
                regions_table33 = regionprops_table(labeled_m33, properties=properties)
                mask_df = pd.DataFrame(regions_table)
                m33_df = pd.DataFrame(regions_table33)
                mask_df['aspect_ratio'] = mask_df['major_axis_length'] / mask_df['minor_axis_length']
                mask_df['equivalent_diameter'] = np.sqrt(4 * mask_df['area'] / np.pi)
                m33_df['aspect_ratio'] = m33_df['major_axis_length'] / m33_df['minor_axis_length']
                m33_df['equivalent_diameter'] = np.sqrt(4 * m33_df['area'] / np.pi)
                mask_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                m33_df.replace([np.inf, -np.inf], np.nan, inplace=True)

                mask_df_drop = mask_df.copy()
                m33_df_drop = m33_df.copy()

                mask_df_drop.dropna(inplace=True)
                m33_df_drop.dropna(inplace=True)

                master_mask_df = pd.concat([master_mask_df, mask_df], ignore_index=True)
                master_mask_drop_df = pd.concat([master_mask_drop_df, mask_df_drop], ignore_index=True)
                master_m33_df = pd.concat([master_m33_df, m33_df], ignore_index=True)
                master_m33_drop_df = pd.concat([master_m33_drop_df, m33_df_drop], ignore_index=True)

        ensure_dir(f'{out_dir}/model_dfs/{model_name}')
        master_mask_df.to_csv(f'{out_dir}/model_dfs/{model_name}/{model_name}_nodrop.csv')
        master_mask_drop_df.to_csv(f'{out_dir}/model_dfs/{model_name}/{model_name}_drop.csv')
        master_m33_df.to_csv(f'{out_dir}/model_dfs/{model_name}/{model_name}_d3_e3_nodrop.csv')
        master_m33_drop_df.to_csv(f'{out_dir}/model_dfs/{model_name}/{model_name}_d3_e3_drop.csv')

    with open(os.path.join(out_dir, f"final_info_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(master_info, f, indent=4)
    return master_info, master_train_log, master_val_log


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
parser.add_argument("--num_epochs", type=int, default=400, help="Number of epochs in each training run")
args = parser.parse_args()


if __name__ == "__main__":
    num_seeds = {
        "training_512": 1,
    }

    out_dir = args.out_dir
    all_results = {}
    final_infos = {}
    for dataset in num_seeds.keys():
        final_info_list = []
        final_info_dict = {}
        for seed_offset in range(num_seeds[dataset]):
            print(f"Running {dataset} with seed offset {seed_offset}")
            final_info, train_info, val_info = run(args.out_dir, dataset, seed_offset, 1)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)
        for model_name in final_info_list[0].keys():
            model_list = []
            for i in range(len(final_info_list)):
                model_list.append(final_info_list[i][model_name])
            final_info_dict[model_name] = model_list
        means = {}
        std_errs = {}
        for model_name, values in final_info_dict.items():
            final_train_losses = np.array([])
            final_val_losses = np.array([])
            for d in values:
                final_train_losses = np.append(final_train_losses, d['final_train_loss'])
                final_val_losses = np.append(final_val_losses, d['final_val_loss'])
            means[model_name] = {'final_train_loss_mean': np.mean(final_train_losses),
                                 'final_val_loss_mean': np.mean(final_val_losses)}
            std_errs[model_name] = {'final_train_loss_mean': np.std(final_train_losses) / len(final_train_losses),
                                 'final_val_loss_mean': np.std(final_val_losses) / len(final_val_losses)}
        final_infos[dataset] = {
            "means": means,
            "stderrs": std_errs,
            "final_info_dict": final_info_dict,
        }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=4)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
