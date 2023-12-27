from torch import nn
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, ConvNeXt
from torchvision.models.convnext import CNBlock
# from timm.models.layers import DropPath
import math
import torch.nn.functional as F
import cv2
import numpy as np
from models.utils import locate_instances

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class CrowdConvNext(nn.Module):

    def __init__(self, pretrained: bool = False, backbone: nn.Module = None, classification_head: bool = False, layers: int = 9):
        super().__init__()
        
        if backbone:
            self.convnext = backbone
        else:
            self.convnext = convnext_tiny(weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)

            if layers < 9:
                self.convnext.features[5] = self.convnext.features[5][:layers]

        self.head = nn.Sequential(nn.Linear(self.convnext.features[-1][-1].block[-2].out_features, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 20 if classification_head else 1))
        
        
        self.classification_head = classification_head

    def build_weight_layer(self):
        with torch.no_grad():
            self.weight_layer = nn.Conv2d(self.head[0].in_features, self.head[0].out_features, kernel_size=1)
            self.weight_layer.weight = nn.Parameter(self.head[0].weight.squeeze(-1).squeeze(-1))
        

    def train_forward(self, x):
        x = self.convnext.features(x)
        x = self.convnext.avgpool(x)
        x = self.head(x.flatten(1))
        return x
        
    def regression_activation_mapping(self, x):
        with torch.no_grad():
            latent = self.convnext.features(x)
            weights = self.weight_layer(latent)
            b, ch, h, w = latent.shape
            dim = ch // 3
            latent = latent.reshape(b, dim, 3, h, w)
            latent = (latent * weights).sum(1)
            return latent
    
    def predict(self, x):
        return torch.max(x, 1)[-1].unsqueeze(-1).float() if self.classification_head else x.round()

    def forward(self, x):
        out = self.train_forward(x)
        return out if self.training else self.predict(out)

    def build_weight_layer(self):
        with torch.no_grad():
            first_head_lin = self.head[0]
            second_head_lin = self.head[-1]
            
            self.weight_layer_1 = nn.Conv2d(first_head_lin.in_features, first_head_lin.out_features, kernel_size=1)
            self.weight_layer_2 = nn.Conv2d(second_head_lin.in_features, second_head_lin.out_features, kernel_size=1)
            
            self.weight_layer_1.weight[:,:,0,0] = first_head_lin.weight[:,:]
            self.weight_layer_2.weight[:,:,0,0] = second_head_lin.weight[:,:]

    def switch_stride(self):
        import copy
        self.detection_convnext = copy.deepcopy(self.convnext)
        weigh_conv = self.detection_convnext.features[6][1]
        new_conv = nn.Conv2d(384, 768, kernel_size=(2, 2), stride=(1, 1))
        with torch.no_grad():
            new_conv.weight[...] = weigh_conv.weight[...]
            new_conv.bias[...] = weigh_conv.bias[...]
        self.detection_convnext.features[6][1] = new_conv

    def locate_people(self, x, num_instances):
        relu = self.head[1]
        latent = self.detection_convnext.features(x)
        latent = self.weight_layer_1(latent)
        latent = self.weight_layer_2(relu(latent))
        im = latent.squeeze(0).permute(1, 2, 0).detach().numpy()
        th = 21
        ind_one = im > th
        ind_zer0 = im <= th
        im[ind_one] = 1
        im[ind_zer0] = 0
        im = im.repeat(3, axis=0).repeat(3, axis=1).astype(np.uint8) # upsample
        contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_contour_area = 0  # Adjust as needed
        max_contour_area = 30000  # Adjust as needed
        filtered_contours= []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_contour_area < area < max_contour_area:
                filtered_contours.append(contour)
        total_minima_region_area = 0
        minimas = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            total_minima_region_area += w*h
            minimas += [{'x':x, 'y':y, 'w':w, 'h':h}]

        if len(minimas) > 0:
            return locate_instances(num_instances, minimas, total_minima_region_area, im_map=im)
        else:
            return [[0.5, 0.5]]

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

def to_cnbv1_2_cnbv2(cnb):
    mods = [] 
    for mod in cnb:
        if isinstance(mod, CNBlock):
            sub_mods = []
            for i, sub_mod in enumerate(mod.block):
                if isinstance(sub_mod, nn.GELU):
                    sub_mods.append(sub_mod)
                    sub_mods.append(GRN(mod.block[i+1].in_features))
                else:
                    sub_mods.append(sub_mod)
            mod.block = nn.Sequential(*sub_mods)
            mods.append(mod)
    
    new_cnb = nn.Sequential(*mods)

    return new_cnb

def restructure_convnextv2(convnext):
    ds = []
    stages = []
    for mod in convnext.features:
        if isinstance(mod[0], CNBlock):
            new_mod = to_cnbv1_2_cnbv2(mod)
            stages.append(new_mod)
        else:
            ds.append(mod)
    
    return stages, ds



class ConvNeXtV2Wrapper(nn.Module):
    def __init__(self, convnextv1):
        super().__init__()
        self.stages, self.downsample_layers = restructure_convnextv2(convnextv1)
        self.convnextv1 = convnextv1

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2).unsqueeze(1)

    def features(self, x, mask = None):
        for i in range(4):
            if mask is not None:
                scale = x.shape[-1] // int(mask.shape[-1] ** 0.5)
                current_mask = self.upsample_mask(mask, scale)
                x *= current_mask
                x = self.downsample_layers[i](x)

                scale = x.shape[-1] // int(mask.shape[-1] ** 0.5)
                current_mask = self.upsample_mask(mask, scale)
                x *= current_mask
                x = self.stages[i](x)
                x *= current_mask
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        return x

    def avg_pool(self, x):
        return self.convnextv1.avgpool(x)

    def forward(self, x, mask = None):
        x = self.features(x, mask)
        x = self.avg_pool(x)
        return x



class MAEConvNext(nn.Module):
    def __init__(self,
                 convnextv1,
                 ratio: float = 0.6,
                 img_size=224,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 decoder_depth=1,
                 decoder_embed_dim=512,
                 patch_size=32,
                 norm_pix_loss=False):
        super().__init__()
        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss


        self.encoder = ConvNeXtV2Wrapper(convnextv1.convnext) if isinstance(convnextv1, CrowdConvNext) else ConvNeXtV2Wrapper(convnextv1)
        self.ratio = ratio

        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)
    
    def mask(self, x, ratio: float = 0.4):
        b = x.shape[0]
        no_of_patches = (x.shape[2] // self.patch_size) ** 2
        uncovered_patches = math.ceil(ratio * no_of_patches)

        ids_shuffle = torch.argsort(torch.randn(b, no_of_patches, device=x.device), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask_ = torch.ones([b, no_of_patches], device=x.device)
        mask_[:, :uncovered_patches] = 0
        mask_ = torch.gather(mask_, dim=1, index=ids_restore)

        return mask_, ids_restore
    
    def forward_encoder(self, imgs, mask):
        x = self.encoder.features(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward(self, x, mask_ratio=0.6):
        mask, _ = self.mask(x, mask_ratio)
        x, mask = self.forward_encoder(x, mask)
        pred = self.forward_decoder(x, mask)
        return pred, mask

    def get_encoder(self, classification_head):
        return CrowdConvNext(self.encoder.convnextv1, classification_head=classification_head)
    
if __name__ == '__main__':
    m = CrowdConvNext()
    m