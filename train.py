from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple,trunc_normal_
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torchvision.utils import save_image
import cv2
import numpy as np
from torchvision.transforms import functional as Fc
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import json
from torch.nn.modules.loss import CrossEntropyLoss
import timm
from utils import DiceLoss
import logging
import torch.nn as nn
from scipy import ndimage
from Former import *
import math
from Global2 import *
from CP_attention import *
from pvtv2 import pvt_v2_b2, pvt_v2_b5, pvt_v2_b0
from decod import CUP, CASCADE, CASCADE_Cat, GCUP, GCUP_Cat, GCASCADE, GCASCADE_Cat
from pyramid_vig2 import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu
from torchvision.models import resnet34


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y
    
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, B):
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


def ct_dewindow(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(-1, W // window_size, H // window_size, window_size, window_size, N).permute(0, 5, 1, 3, 2, 4)
    ct2 = ct2.reshape(bs, N, W * H).transpose(1, 2)
    return ct2


def ct_window(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(bs, H // window_size, window_size, W // window_size, window_size, N)
    ct = ct.permute(0, 1, 3, 2, 4, 5)
    return ct
class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class CrossAtten(nn.Module):

    def __init__(self, in_channels,in_channels2, out_channels, imagesize, depth, patch_size, heads, dim_head=128, dropout=0.1,
                 emb_dropout=0.1):
        # in_chan=128 out_chan=128 img=28 patch=1
        super().__init__()
        image_height, image_width = pair(imagesize)  # 28 28
        patch_height, patch_width = pair(patch_size)  # 1 1
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 28 28
        self.patch_dim = in_channels * patch_height * patch_width  # 64
        self.dmodel = out_channels  # 128
        self.mlp_dim = self.dmodel * 8  # 512
        self.to_patch_embedding_c1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # 196 256 不变 14 14 256
            nn.Linear(in_channels, self.dmodel),
        )
        self.to_patch_embedding_c2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # 196 256 不变 14 14 256
            nn.Linear(in_channels2, self.dmodel),
        )
        # self.to_patch_embedding_c2 = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # b 784 128
        #     nn.Linear(in_channels, self.dmodel),  # [ b 784 128] 不变
        # )

        # dmodel=128 depth=1 heads=4 dim_head=128 mlp_dim=512 num_patch=3136 ? 784
        self.transformer = CPA1(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout,
                                               num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),  # image_height // patch_height
        )

    def forward(self, c1, c2):

        c1 = self.to_patch_embedding_c1(c1)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        
        
        c2 = self.to_patch_embedding_c2(c2)

        # transformer layer
        ax = self.transformer(c1, c2)
        out = self.recover_patch_embedding(ax)
        return out
class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self,
                 window_size,
                 pretrained_window_size,
                 num_heads, seq_length,
                 ct_correct=False,
                 no_log=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)
        self.ct_correct = ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):
        if self.deploy:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            n_global_feature = input_tensor.shape[2] - local_window_size
            if n_global_feature > 0 and self.ct_correct:

                step_for_ct = self.window_size[0] / (n_global_feature ** 0.5 + 1)
                seq_length = int(n_global_feature ** 0.5)
                indices = []
                for i in range(seq_length):
                    for j in range(seq_length):
                        ind = (i + 1) * step_for_ct * self.window_size[0] + (j + 1) * step_for_ct
                        indices.append(int(ind))

                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
                left_part = relative_position_bias[:, :, indices]
            relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                      0,
                                                                                      n_global_feature,
                                                                                      0)).contiguous()
            if n_global_feature > 0 and self.ct_correct:
                relative_position_bias = relative_position_bias * 0.0
                relative_position_bias[:, :n_global_feature, :n_global_feature] = lefttop_part
                relative_position_bias[:, :n_global_feature, n_global_feature:] = top_part
                relative_position_bias[:, n_global_feature:, :n_global_feature] = left_part

            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor
class CMlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x
class HAT(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=7,
                 last=False,
                 layer_scale=None,
                 ct_size=1,
                 do_propagation=False):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim=dim, rank=2, seq_length=4 ** 2)
        self.norm1 = norm_layer(dim)
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size ** 2 if sr_ratio > 1 else 0
        # total number of carrier tokens
        self.cr_window = ct_size
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=80)

        self.drop_path =  nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = 4

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.cr_window = ct_size
        
    
        
    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)
        
        # if self.sr_ratio > 1:
        Bg, Ng, Hg = ct.shape

        
        
      
        x = torch.cat((ct, x), dim=1)
        
        
        
        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4 * self.mlp(self.norm2(x)))
        
        ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
        ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.

        return x,ct
    
    
class HAT3(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=7,
                 last=False,
                 layer_scale=None,
                 ct_size=1,
                 do_propagation=False):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim=dim, rank=2, seq_length=7 ** 2)
        self.norm1 = norm_layer(dim)
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size ** 2 if sr_ratio > 1 else 0
        # total number of carrier tokens
        self.cr_window = ct_size
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=7 ** 2)

        self.drop_path =  nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = 7

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.cr_window = ct_size
        
    
        
    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)
        
        # if self.sr_ratio > 1:
        Bg, Ng, Hg = ct.shape

        
        
      
        x = torch.cat((ct, x), dim=1)
        
        
        
        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4 * self.mlp(self.norm2(x)))
        
        ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
        ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.

        return x,ct
    
class WindowAttention(nn.Module):
    """
    Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 resolution=0,
                 seq_length=0):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            resolution: feature resolution.
            seq_length: sequence length.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(window_size=[resolution, resolution],
                                              pretrained_window_size=[resolution, resolution],
                                              num_heads=num_heads,
                                              seq_length=seq_length)

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.pos_emb_funct(attn, self.resolution ** 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 seq_length=4,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1, bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1, bias=False))
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        # self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not False else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_h -= seq_length // 2
                relative_coords_h /= (seq_length // 2)
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            else:
                seq_length = int(seq_length ** 0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_table = torch.stack(
                    torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not False:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1, 2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
  
        input_tensor = input_tensor + self.pos_emb
        return input_tensor
class HAT2(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=7,
                 last=False,
                 layer_scale=None,
                 ct_size=1,
                 do_propagation=False):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim=dim, rank=2, seq_length=4 ** 2)
        self.norm1 = norm_layer(dim)
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size ** 2 if sr_ratio > 1 else 0
        # total number of carrier tokens
        self.cr_window = ct_size
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=32)

        self.drop_path =  nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = 4

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.cr_window = ct_size
        
    
        
    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)
        
        # if self.sr_ratio > 1:
        Bg, Ng, Hg = ct.shape

        
        
      
        x = torch.cat((ct, x), dim=1)
        
        
        
        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4 * self.mlp(self.norm2(x)))
        
        ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
        ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.

        return x,ct
device = torch.device("cuda")  # 或者选择适当的CUDA设备
def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()
def calculate_similarity_loss(tokens_out):
    # tokens_out has shape (batch_size, 2, feature_dim)
    token1 = tokens_out[:, 0, :]
    token2 = tokens_out[:, 1, :]
    sim_loss = 1 - F.cosine_similarity(token1, token2).mean()  # We subtract from 1 as higher cosine similarity means they are more similar, and we want to minimize this similarity
    return sim_loss
class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False

    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size // 2,
                        groups=self.dim)  # equal to x + conv(x)
class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True,
                 downsample=False, kernel_size=5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
def l2norm(X, dim=-1, eps=1e-12):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -2) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=device, dtype=logits.dtype),
        torch.tensor(1., device=device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


class Fusion(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, out_dim, bias=False) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim_1, out_dim, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
#         if in_1.shape[-1] < in_2.shape[-1]:
#             in_1 = F.interpolate(in_1, size=in_2.shape[-2:], mode='bilinear', align_corners=True)
#         elif in_1.shape[-1] > in_2.shape[-1]:
#             in_2 = F.interpolate(in_2, size=in_1.shape[-2:], mode='bilinear', align_corners=True)

#         x = torch.cat((in_1, in_2), dim=1)
        x = self.fusion(x)
        return x
# class Fusion(nn.Module):
#     def __init__(self, in_dim_1, out_dim, bias=False) -> None:
#         super().__init__()
#
#         self.fusion = nn.Sequential(
#             nn.Conv2d(in_dim_1, out_dim, 3, padding=1, bias=bias),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(),
#             nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=bias),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(),
#         )
#         self.in_dim_1 = in_dim_1
#         self.out_dim = out_dim
#
#     def forward(self, in_1):
#         print(self.out_dim)
#         print(self.in_dim_1)
#         x = self.fusion(in_1)
#         return x





class DProjector(nn.Module):
    def __init__(self, text_dim=512, in_dim=512, kernel_size=1):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector

        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample((415, 375), mode='bilinear', align_corners=True),
            conv_layer(in_dim, in_dim, 3, padding=1),
            
            nn.Conv2d(in_dim, in_dim, 1))

        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(text_dim, out_dim)
        self.final_super_0_4 = nn.Sequential(
            nn.Upsample((415, 375), mode='bilinear', align_corners=True),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )
    def forward(self, x, text):
        '''
            x: b, 512, 104, 104
            text: b, 512
        '''
        x = self.vis(x)  # Eq. 8

        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, 1, (256*3*3 + 1) -> b, 1, 256, 3, 3 / b
        text = self.txt(text)  # Eq. 8

        weight, bias = text[:, :-1], text[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=0,
                       groups=B,
                       bias=bias)

        # b, 1, 104, 104
        out = out.transpose(0, 1)

 

        return out


class CrossAttn(nn.Module):
    def __init__(self,
                 q_dim,
                 kv_dim,
                 hidden_dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = q_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_fuse = qkv_fuse

        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value=None, mask=None):
        B, N, C = query.shape
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, N, S]

        if mask is not None:
            mask = mask[:, None, :, None].expand(-1, self.num_heads, -1, -1)  # b nh S 1
            k = k * mask
            v = v * mask
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn + (1e4 * mask.transpose(-2, -1) - 1e4)  # b nh 1 S
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        assert attn.shape == (B, self.num_heads, N, S)
        # [B, nh, N, C//nh] -> [B, N, C]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class OriLoadToken(nn.Module):
    def __init__(self, token_dim, bias, drop) -> None:
        super().__init__()
        self.cross_attn = CrossAttn(
            q_dim=token_dim,
            kv_dim=768,
            hidden_dim=token_dim,
            num_heads=1,
            out_dim=token_dim,
            qkv_bias=bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.normq = nn.LayerNorm(token_dim)
        self.normk = nn.LayerNorm(768)

        self.normq = nn.LayerNorm(token_dim)
        self.normk = nn.LayerNorm(768)

    def forward(self, tokens, text, pad_mask):
        tokens = tokens + self.cross_attn(query=self.normq(tokens), key=self.normk(text.permute(0, 2, 1)),
                                          mask=pad_mask[..., 0])
        return tokens


# updated version
class LoadToken(nn.Module):
    def __init__(self, token_dim, bias, drop) -> None:
        super().__init__()
        self.cross_attn = CrossAttn(
            q_dim=token_dim,
            kv_dim=768,
            hidden_dim=token_dim,
            num_heads=1,
            out_dim=token_dim,
            qkv_bias=bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.normq = nn.LayerNorm(token_dim)
        self.normk = nn.LayerNorm(768)
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = Mlp(token_dim, token_dim * 2, token_dim)

    def forward(self, tokens):
        self.to(device)
        ltoken, ttoken = torch.split(tokens, [tokens.shape[1] - 1, 1], dim=1)

        tokens = torch.cat((ltoken, ttoken), dim=1)
        return tokens


class LoadLayer(nn.Module):
    def __init__(self, token_dim, drop, bias=False, pe_shape=None) -> None:
        super().__init__()
        if pe_shape > 30:
            self.loadtoken = LoadToken(
                token_dim=token_dim,
                bias=bias,
                drop=drop
            )
            self.norm = nn.LayerNorm(token_dim)
            self.mlp = Mlp(token_dim, token_dim * 2, token_dim)
        self.positional_embedding = nn.Parameter(torch.randn(pe_shape ** 2, token_dim) / token_dim ** 0.5)
        self.pe_shape = pe_shape

    def forward(self, tokens):
        self.to(device)

        if self.pe_shape > 30:
            tokens = self.mlp(self.norm(tokens))
        return tokens, self.positional_embedding


class CGAttention(nn.Module):
    def __init__(self, token_dim, vis_dim, hidden_dim, drop=0., bias=True) -> None:
        super().__init__()
        self.norm_v = nn.LayerNorm(vis_dim)
        self.norm_t = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(vis_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(vis_dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, token_dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = Mlp(token_dim, token_dim * 2, token_dim, drop=drop)
        self.tau = nn.Parameter(torch.ones(1), requires_grad=True)

    def with_pe(self, vis, pe):
        return vis + pe

    def forward(self, tokens, vis, pe=None):
       
        b, c, h, w = vis.shape
        vis = rearrange(vis, 'b c h w -> b (h w) c')
        if pe is not None:
            vis = self.with_pe(vis, pe)
        vis = self.norm_v(vis)
        q = self.q_proj(self.norm_t(tokens))
        k = self.k_proj(vis)
        v = self.v_proj(vis)

        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)
        raw_attn = (q @ k.transpose(-2, -1))
        tau = torch.clamp(self.tau, max=0).exp()
        attn = gumbel_softmax(raw_attn, dim=-2, tau=tau)
        hit_map = attn
        # 8,2,900

        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1)
        new_tokens = attn @ v
        new_tokens = self.proj_drop(self.proj(new_tokens))
        new_tokens = self.mlp(self.norm(new_tokens + tokens))
        return new_tokens, hit_map.reshape(b, -1, h, w)



def ellipse_mask(h, w, center=None, r1=None, r2=None):
    """生成一个椭圆mask。"""
    if center is None:  # 默认为图像中心
        center = (h // 2, w // 2)
    if r1 is None:  # 默认为图像高度的一半
        r1 = h // 2
    if r2 is None:  # 默认为图像宽度的一半
        r2 = w // 2

    Y, X = np.ogrid[:h, :w]
    dist = (X - center[1]) ** 2 / r2 ** 2 + (Y - center[0]) ** 2 / r1 ** 2
    mask = dist <= 1
    return mask


class MaskedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        seed = np.random.randint(2147483647)  # 为了使mask和image在随机变换后保持对应关系，需要固定一个随机种子
        random.seed(seed)
        image = self.transform(image)
        random.seed(seed)
        mask = self.transform(mask)
        return image, mask


# 2. 添加椭圆mask到图像的函数
def apply_ellipse_mask(img, mask):
    """应用椭圆mask到图像上。"""
    if not Fc._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img_np = np.array(img)
    mask_broadcasted = np.broadcast_to(mask[:, :, None], img_np.shape)
    img_np[~mask_broadcasted] = 0  # 把不在椭圆内的部分设置为0
    return Fc.to_pil_image(img_np)


# 3. 添加到您的数据增强流程中
class EllipseMaskTransform:
    def __init__(self, h, w, center=None, r1=None, r2=None):
        self.mask = ellipse_mask(h, w, center, r1, r2)

    def __call__(self, img):
        img = img.resize((self.mask.shape[1], self.mask.shape[0]))  # 确保图像大小与mask匹配
        return apply_ellipse_mask(img, self.mask)

# 自定义添加噪声的变换（如果需要）
class AddNoiseTransform:
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, img):
        noise = torch.randn(img.size()) * self.noise_level
        return img + noise

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mtransform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mtransform = mtransform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
      
        mask = Image.fromarray(mask)
        # mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
            mask = self.mtransform(mask)
        return image, mask

class AddNoiseTransform:
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, img):
        # 确保图像是一个Tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.functional.to_tensor(img)
        # 生成和图像尺寸相同的噪声
        noise = torch.randn(img.shape) * self.noise_level
        noisy_img = img + noise
        # 返回值应当被裁剪在[0, 1]范围内
        return torch.clamp(noisy_img, 0, 1)
    
    
    
    
transform = transforms.Compose([
    transforms.Resize((375, 375)),
    transforms.ToTensor(),
])
mtransform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = CustomDataset('/root/autodl-tmp/area/croimage', '/root/autodl-tmp/area/cromask', transform=transform, mtransform=mtransform)

testdataset = CustomDataset('/root/autodl-tmp/tarea1/croimage', '/root/autodl-tmp/tarea1/cromask', transform=transform, mtransform=mtransform)

test2dataset = CustomDataset('/root/autodl-tmp/tarea2/croimage', '/root/autodl-tmp/tarea2/cromask', transform=transform, mtransform=mtransform)

test3dataset = CustomDataset('/root/autodl-tmp/tt4/croimage', '/root/autodl-tmp/tt4/cromask', transform=transform, mtransform=mtransform)

val_dataset = CustomDataset('/root/autodl-tmp/val/croimage', '/root/autodl-tmp/val/cromask', transform=transform, mtransform=mtransform)
# # 定义验证集的大小。
# val_size = 327 
# train_size = len(dataset) - val_size

# # 创建训练集和验证集的索引。
# train_indices = list(range(0, train_size))
# val_indices = list(range(train_size, len(dataset)))

# 使用这些索引创建训练集和验证集。
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)


# testdataset = CustomDataset('/root/autodl-tmp/area/cropped', '/root/autodl-tmp/area/cromask', transform=transform)
# dataset = CustomDataset('/root/autodl-tmp/tarea/croimage', '/root/autodl-tmp/tarea/cromask', transform=transform)
# combined_dataset = ConcatDataset([testdataset, dataset])
# train_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

# # Splitting the dataset
# train_size = int(0.94 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset2, val_dataset = random_split(dataset, [train_size, val_size])

# # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# test_loader = DataLoader(testdataset, batch_size=1, shuffle=False)




# dataset_size = len(dataset)

# # 确保 train_size 和 val_size 的总和不超过 dataset_size

# val_size = 327 # 设定验证集大小
# train_dataset = dataset[:dataset_size-val_size]
# val_dataset = dataset[dataset_size-val_size:dataset_size]
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=24, shuffle=False)
test2_loader = DataLoader(test2dataset, batch_size=24, shuffle=False)
test3_loader = DataLoader(test3dataset, batch_size=10, shuffle=False)
def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[
           :,
           :,
           delta: tensor_size - delta if delta * 2 == tensor_size - target_size else tensor_size - (delta + 1),
           delta: tensor_size - delta if delta * 2 == tensor_size - target_size else tensor_size - (delta + 1),
           ]


class OBLNet(nn.Module):
    def __init__(self, n_class=1, img_size=375, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(OBLNet, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/root/autodl-tmp/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(256, self.n_class, 1)
        self.out_head2 = nn.Conv2d(256, self.n_class, 1)
        self.out_head3 = nn.Conv2d(192, self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        
        self.out_head5 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head6 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head7 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head8 = nn.Conv2d(self.channels[3], self.n_class, 1)
        cur = 0
     
        self.scale = 2
        factor = 2
        self.window_size = 7
        
        
     
        
        
        
        
   
        
        
        
        dropout = 0.
        self.to_out1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0)
            
            # nn.Dropout(dropout)
        )
        self.to_out2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, padding=0),
            nn.Dropout(dropout)
        )
        self.to_out3 = nn.Sequential(
            nn.Conv2d(448, 320, kernel_size=1, padding=0),
            # nn.Linear(1024, 256),
            nn.Dropout(dropout)
        )
        self.to_out4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            # nn.Linear(512 * 3, 512),
            nn.Dropout(dropout)
        )
        self.norm11 = nn.BatchNorm2d(896)
        self.norm22 = nn.BatchNorm2d(896)
        self.norm33 = nn.BatchNorm2d(1024)
        self.norm44 = nn.BatchNorm2d(512*3)
        
        self.norm5 = nn.BatchNorm2d(64)
        self.norm6 = nn.BatchNorm2d(128)
        self.norm7 = nn.BatchNorm2d(320)
        self.norm8 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout(0.1)

        # self.transformer3 = CrossAtten(in_channels=256 // factor // self.scale ,
        #                                               out_channels=256 // factor // self.scale ,
        #                                               imagesize=img_size // 4, depth=1, heads=2, patch_size=1)
        self.up1 = nn.ConvTranspose2d(512, 320, kernel_size=2, stride=2, padding=0)
        self.up2 = nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2, padding=0)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.up4 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=0)
        
        
        self.window_size = 4
        self.blocks3 = nn.ModuleList([
                HAT3(dim=320,
                    num_heads=8,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=nn.Identity(),
                    sr_ratio=1.,
                    window_size=7,
                    last=(i == 6 - 1),
                    layer_scale=1e-5,
                    ct_size=2,
                    do_propagation=True,
                    )
                for i in range(3)])

        
        
        self.blocks4 = nn.ModuleList([
                HAT3(dim=512,
                    num_heads=8,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=nn.Identity(),
                    sr_ratio=1.,
                    window_size=7,
                    last=(i == 6 - 1),
                    layer_scale=1e-5,
                    ct_size=2,
                    do_propagation=True,
                    )
                for i in range(3)])

        
        
        
        
        
        self.blocks2 = nn.ModuleList([
                HAT(dim=128,
                    num_heads=8,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=nn.Identity(),
                    sr_ratio=1.,
                    window_size=7,
                    last=(i == 6 - 1),
                    layer_scale=1e-5,
                    ct_size=2,
                    do_propagation=True,
                    )
                for i in range(3)])

        
        
        
        
        
        self.blocks1 = nn.ModuleList([
                HAT2(dim=256,
                    num_heads=8,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=nn.Identity(),
                    sr_ratio=1.,
                    window_size=7,
                    last=(i == 6 - 1),
                    layer_scale=1e-5,
                    ct_size=2,
                    do_propagation=True,
                    )
                for i in range(3)])

        
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool = resnet.maxpool
        
        
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        
        
        
        dim = 320

        self.decoder1 = DecoderBottleneckLayer(512)
        self.decoder2 = DecoderBottleneckLayer(512)
        self.decoder3 = DecoderBottleneckLayer(384)
        self.up3_1 = nn.ConvTranspose2d(512,256, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        self.up1_1 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=4)
        # dim = 320
        # self.SE_0 = SEBlock(128)
        self.SE_1 = SEBlock(512)
        self.SE_2 = SEBlock(512)
        self.SE_3 = SEBlock(384)
        # self.SE_4 = SEBlock(128)
        # # self.decoder1 = DecoderBottleneckLayer(4*dim + 512)
        # self.decoder1 = DecoderBottleneckLayer(512)
        # self.decoder2 = DecoderBottleneckLayer(640)
        # self.decoder3 = DecoderBottleneckLayer(384)
        # self.decoder4 = DecoderBottleneckLayer(192)
        # self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.up2_1 = nn.ConvTranspose2d(640, 256, kernel_size=2, stride=2)
        # self.up1_1 = nn.ConvTranspose2d(384, 128, kernel_size=2, stride=2)
        # self.up4_1 = nn.ConvTranspose2d(192, 64, kernel_size=4, stride=4)
        
        
        self.to_patch_embedding_c0 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w ', p1=2, p2=2)
        )

        self.re_patch_embedding_c0 = nn.Sequential(
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=64)
        )
        self.out = nn.Conv2d(192, self.n_class,kernel_size=1)
        
        
        self.final_super_0_4 = nn.Sequential(
            nn.Upsample((415, 375), mode='bilinear', align_corners=True),
        )
        
        self.final_super_0_3 = nn.Sequential(
            nn.Upsample((415, 375), mode='bilinear', align_corners=True),
           
        )
    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        
        
        
        
        
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        
        out3 = x3
        e4 = e4
        
#         # B2, C2, H2, W2 = x2.shape
#         out3 = self.transformer1(x3,e4)
#         e4 = self.transformer2(e4, out3)
       
        
  
        # 使用双向cross attention
        B2, C2, H2, W2 = x2.shape
        eB2, eC2, eH2, eW2 = e2.shape
        out2 = window_partition(x2, 4)
        ec2 = window_partition(e2, 8)
        for bn, (blk) in enumerate(self.blocks2):
            out2,ec2 = blk(out2,ec2)
            # out3 = window_reverse(out3, self.window_size, H3, W3, B3)
        out2 = window_reverse(out2, 4, H2, W2, B2)


        ec2 = window_reverse(ec2, 8, eH2, eW2, eB2)
        
        
        
        x1 = self.to_patch_embedding_c0(x1)
        
        B1, C1, H1, W1 = x1.shape
        ec3 = window_partition(e3, self.window_size)
        out1 = window_partition(x1, self.window_size)
        for bn, (blk) in enumerate(self.blocks1):
            out1,ec3 = blk(out1,ec3)
            # out1 = window_reverse(out1, self.window_size, H1, W1, B1)
        out1 = window_reverse(out1, self.window_size, H1, W1, B1)
        ec3 = window_reverse(ec3, self.window_size, H1, W1, B1)
        
        out1 = self.re_patch_embedding_c0(out1)
        
        
        
        
        
        
        
        
        e4 = self.SE_1(e4)
        cat_1 = self.decoder1(e4)
        cat_1 = self.up3_1(cat_1)

        # cat_2 = torch.cat([v2_cnn, e3], dim=1)
        # cat_2 = self.SE_2(cat_2)
        cat_2 = torch.cat([ec3, cat_1],dim=1)
        
     
        cat_2 = self.SE_2(cat_2)
        cat_2 = self.decoder2(cat_2)
        cat_2 = self.up2_1(cat_2)

        # cat_3 = torch.cat([v1_cnn, e2], dim=1)
        # cat_3 = self.SE_3(cat_3)
        cat_3 = torch.cat([ec2, cat_2], dim=1)
  
        cat_3 = self.SE_3(cat_3)
        cat_3 = self.decoder3(cat_3)
        cat_3 = self.up1_1(cat_3)
        out = self.out(cat_3)
        out = self.final_super_0_3(out)  
        out4 = x4
        # y4 = x4

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(out4, [out3, out2, out1])

        
        
        
        p4 = self.out_head8(x4_o)  
   
    
  
        # p3 = F.interpolate(p3, scale_factor=4, mode='bilinear')
        
        p4 = self.final_super_0_3(p4)  
        
      
 
        return out,p4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OBLNet(n_class=1, img_size=375, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation='additive').to(device)
model_save_dir2 = './best_modelrerere6.pth'
# state_dict = torch.load('./best_modelrerere6.pth')

# model.load_state_dict(state_dict)
# model = Decoder().to(device)
# model = UNet().to(device)
# 1. 加载预训练模型

# model = UNetPlusPlus(n_channels=3, n_classes=2).to(device)  # 可以根据需要调整 n_classes
optimizer = optim.Adam(model.parameters(), lr=0.0001)
weight = torch.Tensor([2]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

lowest_val = float('inf')
lowest_test = float('inf')
lowest_test2 = float('inf')
save_path = './best_modelrerere6.pth'
save_pathval1 = './best_modelrerereval1.pth'
save_pathval = './best_modelrerereval.pth'
save_path3 = './best_modelrerere3.pth'
save_path2 = './best_modelrerere2.pth'
save_path1 = './best_modelrerere1.pth'
epochs = 1000
file_path = 'output_logs5.txt'  # 可以根据需要更改文件名和路径

def save_images(images, epoch, save_path):
    for i, image in enumerate(images):
        # Convert numpy array to image (using PIL or similar library)
        # Image saving logic here
        img = Image.fromarray(np.uint8(image * 255))  # Assuming image is in the range [0, 1]
        img.save(f"{save_path}/img_{i}.bmp")
        
best_images_per_epoch = []
tbest_images_per_epoch = []
t2best_images_per_epoch = []

ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(1)
for epoch in range(epochs):
    model.train()
    loss = 0.
    test3_loss = 0.0
    for images, masks in train_loader:

        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        # out,x_c3,x_c4, maps,tokensst
        x1,x2 = model(images)
     
        
    
        loss =criterion(x1, masks)+ criterion(x2, masks) 
  
        loss.backward()

        # loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(loss)
    with torch.no_grad():
        print('-------------------------------------')
        for images, masks in test3_loader:
            images, masks = images.to(device), masks.to(device)
            outputs,reout = model(images)
            sigmoid_outputs = torch.sigmoid(outputs)  # 使用Sigmoid激活函数
            # loss = criterion(sigmoid_outputs, masks)
            # loss =criterion(outputs, masks)+ criterion(reout, masks)
            # loss = dice_loss(sigmoid_outputs, masks) + sigmoid_focal_loss(sigmoid_outputs, masks, alpha=-1, gamma=0)
            # test3_loss += loss.item()
            outputs = (sigmoid_outputs > 0.5).float()  # 使用0.5作为阈值
            # average_loss = loss.item() / images.size(0) 
            for idx,(img, mask, out) in enumerate(zip(images, masks, outputs)):
              
                save_image(img, f'/root/autodl-tmp/oimg/epoch_{epoch}_img_{idx}.bmp')
                save_image(mask, f'/root/autodl-tmp/omakt/epoch_{epoch}_mask_{idx}.bmp')
                save_image(out, f'/root/autodl-tmp/oout/epoch_{epoch}_output_{idx}.bmp')
#     val_accuracy_list = []
#     val_iou_list = []
#     val_dice_list = []
#     val_precision_list = []
#     val_recall_list = []
    
    
#     test_accuracy_list = []
#     test_iou_list = []
#     test_dice_list = []
#     test_precision_list = []
#     test_recall_list = []
    
    
#     test2_accuracy_list = []
#     test2_iou_list = []
#     test2_dice_list = []
#     test2_precision_list = []
#     test2_recall_list = []
#     model.eval()
#     test_loss = 0.0
#     test2_loss = 0.0
#     test3_loss = 0.0
#     val_loss = 0.0
#     best_val = None
#     best_test = None
#     best_test2 = None
#     best_test3 = None
#     images_with_loss = []
#     images_with_tloss = []
#     images_with_t2loss = []
#     images_with_t3loss = []
#     with torch.no_grad():
#         for images, masks in val_loader:
#             images, masks = images.to(device), masks.to(device)
#             outputs,reout = model(images)
#             sigmoid_outputs = torch.sigmoid(outputs)  # 使用Sigmoid激活函数
#             # loss = criterion(sigmoid_outputs, masks)    
#             loss =criterion(outputs, masks) + criterion(reout, masks)


#             # loss = dice_loss(sigmoid_outputs, masks) + sigmoid_focal_loss(sigmoid_outputs, masks, alpha=-1, gamma=0)
#             val_loss += loss.item()
#             outputs = (sigmoid_outputs > 0.5).float()  # 使用0.5作为阈值
#             average_loss = loss.item() / images.size(0) 
#             # for img, mask, out in zip(images, masks, outputs):
#             #     # 将每张图片及其mask和平均loss存储在列表中
#             #     images_with_loss.append((average_loss, img.cpu(), mask.cpu(), out.cpu()))
 

            
            
            
#             # 将 tensors 转为 numpy arrays 用于评估
#             outputs_np = outputs.cpu().numpy().flatten().astype(int)
#             masks_np = masks.cpu().numpy().flatten().astype(int)
  
            
#             # 计算各项指标
#             accuracy = accuracy_score(masks_np, outputs_np)
#             val_accuracy_list.append(accuracy)

#             iou = jaccard_score(masks_np, outputs_np)
#             val_iou_list.append(iou)

#             dice = f1_score(masks_np, outputs_np)
#             val_dice_list.append(dice)

#             precision = precision_score(masks_np, outputs_np)
#             val_precision_list.append(precision)

#             recall = recall_score(masks_np, outputs_np)
#             val_recall_list.append(recall)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             # 对列表按loss进行排序
# #             images_with_loss.sort(key=lambda x: x[0])

# #             # 保存每个epoch的最佳20张图片
# #             best_20 = images_with_loss[:20]
# #             best_images_per_epoch.append(best_20)

# #             for idx, (_, img, mask, out) in enumerate(best_20):
# #                 save_image(img, f'/root/autodl-tmp/makv5/epoch_{epoch}_img_{idx}.bmp')
# #                 save_image(mask, f'/root/autodl-tmp/makv5/epoch_{epoch}_mask_{idx}.bmp')
# #                 save_image(out, f'/root/autodl-tmp/makv5/epoch_{epoch}_output_{idx}.bmp')
# #         images_with_loss.sort(key=lambda x: x[0])
# #         # 保存每个epoch的最佳20张图片v
# #         best_20 = images_with_loss[:20]
# #         best_images_per_epoch.append(best_20)

# #         for idx, (_, img, mask, out) in enumerate(best_20):
# #             save_image(img, f'/root/autodl-tmp/makv/epoch_{epoch}_img_{idx}.bmp')
# #             save_image(mask, f'/root/autodl-tmp/makv/epoch_{epoch}_mask_{idx}.bmp')
# #             save_image(out, f'/root/autodl-tmp/makv/epoch_{epoch}_output_{idx}.bmp')
            
            
            
#         if val_loss < lowest_val:
#             lowest_val = val_loss
#             val_lowest_loss = epoch
#             # images_test_save = images.cpu().detach().numpy()  # Save the images from the epoch with the lowest loss
#             torch.save(model.state_dict(), save_pathval)

# #         if epoch == val_lowest_loss:
# #             # save_image(torch.sigmoid(outputs), './test_best_outputrerere8.bmp')
            
            
# #             with torch.no_grad():
# #                 for i, (images, masks) in enumerate(val_loader):
# #                     images, masks = images.to(device), masks.to(device)
# #                     outputs,  _,_, _, _,  _,  _= model(images)
# #                     outputs = torch.sigmoid(outputs) > 0.5
# #                     save_image(torch.sigmoid(outputs), f'/root/autodl-tmp/vmak/img_{i}.bmp')
#                     # save_image(torch.sigmoid(outputs), '/root/autodl-tmp/vmak/img_{i}.bmp')
# #                     images_np = images.cpu().numpy()  # 转换为numpy数组
# #                     for j, image in enumerate(images_np):
# #                         save_image(torch.sigmoid(outputs), './test_best_outputrerere8.bmp')

# #                         img = Image.fromarray(np.uint8(image.squeeze() * 255))
# #                         img.save(f"{'/root/autodl-tmp/vmask/'}/epoch_{epoch}_batch_{i}_img_{j}.bmp")
            
#     with torch.no_grad():
#         for images, masks in test2_loader:
#             images, masks = images.to(device), masks.to(device)
#             outputs,reout = model(images)

#             sigmoid_outputs = torch.sigmoid(outputs)  # 使用Sigmoid激活函数
#             # loss = criterion(sigmoid_outputs, masks)
#             loss =criterion(outputs, masks)+ criterion(reout, masks)


#             # loss = dice_loss(sigmoid_outputs, masks) + sigmoid_focal_loss(sigmoid_outputs, masks, alpha=-1, gamma=0)
#             test2_loss += loss.item()
#             outputs = (sigmoid_outputs > 0.5).float()  # 使用0.5作为阈值
#             average_loss = loss.item() / images.size(0) 
#             # for img, mask, out in zip(images, masks, outputs):
#             #     # 将每张图片及其mask和平均loss存储在列表中
#             #     images_with_t2loss.append((average_loss, img.cpu(), mask.cpu(), out.cpu()))

#             # # 遍历这个batch的每张图片和对应的loss
#             # for img, mask, out, l in zip(images, masks, outputs, loss):
#             #     # 将每张图片及其mask和loss存储在列表中
#             #     images_with_t2loss.append((l.item(), img.cpu(), mask.cpu(), out.cpu()))
            
            
            
#             # 将 tensors 转为 numpy arrays 用于评估
#             outputs_np = outputs.cpu().numpy().flatten().astype(int)
#             masks_np = masks.cpu().numpy().flatten().astype(int)

#             # 计算各项指标
#             accuracy = accuracy_score(masks_np, outputs_np)
#             test2_accuracy_list.append(accuracy)

#             iou = jaccard_score(masks_np, outputs_np)
#             test2_iou_list.append(iou)

#             dice = f1_score(masks_np, outputs_np)
#             test2_dice_list.append(dice)

#             precision = precision_score(masks_np, outputs_np)
#             test2_precision_list.append(precision)

#             recall = recall_score(masks_np, outputs_np)
#             test2_recall_list.append(recall)
# #         # 对列表按loss进行排序t
# #         images_with_t2loss.sort(key=lambda x: x[0])

# #         # 保存每个epoch的最佳20张图片
# #         t2best_20 = images_with_t2loss[:20]
# #         t2best_images_per_epoch.append(t2best_20)

# #         for idx, (_, img, mask, out) in enumerate(t2best_20):
# #             save_image(img, f'/root/autodl-tmp/makt2/epoch_{epoch}_img_{idx}.bmp')
# #             save_image(mask, f'/root/autodl-tmp/makt2/epoch_{epoch}_mask_{idx}.bmp')
# #             save_image(out, f'/root/autodl-tmp/makt2/epoch_{epoch}_output_{idx}.bmp')
#         if test2_loss < lowest_test2:
#             lowest_test2 = test2_loss
#             test2_lowest_loss = epoch
#             # images_test_save = images.cpu().detach().numpy()  # Save the images from the epoch with the lowest loss
#             torch.save(model.state_dict(), save_path2)

# #         if epoch == test2_lowest_loss:
# #             # save_image(torch.sigmoid(outputs), './test_best_outputrerere8.bmp')
            
            
# #             with torch.no_grad():
# #                 for i, (images, masks) in enumerate(test2_loader):
# #                     images, masks = images.to(device), masks.to(device)
# #                     outputs, _, _, _, _, _ , _ = model(images)
# #                     outputs = torch.sigmoid(outputs) > 0.5
# #                     save_image(torch.sigmoid(outputs), f'/root/autodl-tmp/tmask2/img_{i}.bmp')
                    
                    
                    
                    
            
            
#     with torch.no_grad():
#         for images, masks in test_loader:
#             images, masks = images.to(device), masks.to(device)
#             outputs,reout = model(images)

#             sigmoid_outputs = torch.sigmoid(outputs)  # 使用Sigmoid激活函数
#             # loss = criterion(sigmoid_outputs, masks)
#             loss =criterion(outputs, masks)+ criterion(reout, masks)


#             # loss = dice_loss(sigmoid_outputs, masks) + sigmoid_focal_loss(sigmoid_outputs, masks, alpha=-1, gamma=0)
#             test_loss += loss.item()
#             outputs = (sigmoid_outputs > 0.5).float()  # 使用0.5作为阈值
#             average_loss = loss.item() / images.size(0) 
#             # for img, mask, out in zip(images, masks, outputs):
#             #     # 将每张图片及其mask和平均loss存储在列表中
#             #     images_with_tloss.append((average_loss, img.cpu(), mask.cpu(), out.cpu()))

#             # # 遍历这个batch的每张图片和对应的loss
#             # for img, mask, out, l in zip(images, masks, outputs, loss):
#             #     # 将每张图片及其mask和loss存储在列表中
#             #     images_with_tloss.append((l.item(), img.cpu(), mask.cpu(), out.cpu()))
            
            
            
#             # 将 tensors 转为 numpy arrays 用于评估
#             outputs_np = outputs.cpu().numpy().flatten().astype(int)
#             masks_np = masks.cpu().numpy().flatten().astype(int)

#             # 计算各项指标
#             accuracy = accuracy_score(masks_np, outputs_np)
#             test_accuracy_list.append(accuracy)

#             iou = jaccard_score(masks_np, outputs_np)
#             test_iou_list.append(iou)

#             dice = f1_score(masks_np, outputs_np)
#             test_dice_list.append(dice)

#             precision = precision_score(masks_np, outputs_np)
#             test_precision_list.append(precision)

#             recall = recall_score(masks_np, outputs_np)
#             test_recall_list.append(recall)
# #         # 对列表按loss进行排序t
# #         images_with_tloss.sort(key=lambda x: x[0])

# #         # 保存每个epoch的最佳20张图片
# #         tbest_20 = images_with_tloss[:20]
# #         tbest_images_per_epoch.append(tbest_20)

# #         for idx, (_, img, mask, out) in enumerate(tbest_20):
# #             save_image(img, f'/root/autodl-tmp/makt5/epoch_{epoch}_img_{idx}.bmp')
# #             save_image(mask, f'/root/autodl-tmp/makt5/epoch_{epoch}_mask_{idx}.bmp')
# #             save_image(out, f'/root/autodl-tmp/makt5/epoch_{epoch}_output_{idx}.bmp')
#         if test_loss < lowest_test:
#             lowest_test = test_loss
#             test_lowest_loss = epoch
#             # images_test_save = images.cpu().detach().numpy()  # Save the images from the epoch with the lowest loss
#             torch.save(model.state_dict(), save_path1)

# #         if epoch == test_lowest_loss:
# #             # save_image(torch.sigmoid(outputs), './test_best_outputrerere8.bmp')
            
            
# #             with torch.no_grad():
# #                 for i, (images, masks) in enumerate(test_loader):
# #                     images, masks = images.to(device), masks.to(device)
# #                     outputs, _, _, _, _, _ , _ = model(images)
# #                     outputs = torch.sigmoid(outputs) > 0.5
# #                     save_image(torch.sigmoid(outputs), f'/root/autodl-tmp/tmask/img_{i}.bmp')
#                     # save_image(torch.sigmoid(outputs), '/root/autodl-tmp/tmask/img_{i}.bmp')
#                     # images_np = images.cpu().numpy()  # 转换为numpy数组
#                     # for j, image in enumerate(images_np):
#                     #     img = Image.fromarray(np.uint8(image.squeeze() * 255))
#                     #     img.save(f"{'/root/autodl-tmp/tmask/'}/epoch_{epoch}_batch_{i}_img_{j}.bmp")
#             # save_images(images_test_save, epoch, '/root/autodl-tmp/tmask/')