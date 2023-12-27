from torchvision.models import vit_b_16, VisionTransformer, ViT_B_16_Weights
from torch import nn
import torch
import math

def process_input(patch_size, dim, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = patch_size
    
    n_h = h // p
    n_w = w // p


    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x

class SmallCrowdViT(nn.Module):

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.vit = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.encoder.layers = self.vit.encoder.layers[:1]

        self.head = nn.Sequential(nn.Linear(self.vit.encoder.layers[-1].mlp[-2].out_features, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 1))

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # TODO Try with class token

        return self.head(self.vit.encoder(x)[:, 0])

class CrowdViT(nn.Module):

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.vit = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.encoder.layers = self.vit.encoder.layers[:2]

        self.head = nn.Sequential(nn.Linear(self.vit.encoder.layers[-1].mlp[-2].out_features, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 1))

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # TODO Try with class token

        return self.head(self.vit.encoder(x)[:, 0])

class BigCrowdViT(nn.Module):

    def __init__(self, pretrained: bool = False, classification_head: bool = False, backbone = None):
        super().__init__()
        if backbone:
            self.vit = backbone
        else :
            self.vit = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.vit.encoder.layers = self.vit.encoder.layers[:3]

        self.head = nn.Sequential(nn.Linear(self.vit.encoder.layers[-1].mlp[-2].out_features, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 20 if classification_head else 1))
        self.classification_head = classification_head

    def train_forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return self.head(self.vit.encoder(x)[:, 0])
    
    def predict(self, x):
        return torch.max(x, 1)[-1].unsqueeze(-1).float() if self.classification_head else x.round()

    def forward(self, x):
        out = self.train_forward(x)
        return out if self.training else self.predict(out)

class VeryBigCrowdViT(nn.Module):

    def __init__(self, pretrained: bool = False, classification_head: bool = False, backbone = None, layers: int = 4):
        super().__init__()
        if backbone:
            self.vit = backbone
        else :
            self.vit = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.vit.encoder.layers = self.vit.encoder.layers[:layers]

        self.head = nn.Sequential(nn.Linear(self.vit.encoder.layers[-1].mlp[-2].out_features, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 20 if classification_head else 1))
        self.classification_head = classification_head

    def train_forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return self.head(self.vit.encoder(x)[:, 0])
    
    def predict(self, x):
        return torch.max(x, 1)[-1].unsqueeze(-1).float() if self.classification_head else x.round()

    def forward(self, x):
        out = self.train_forward(x)
        return out if self.training else self.predict(out)

class FullCrowdViT(nn.Module):

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.vit = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

        self.head = nn.Sequential(nn.Linear(self.vit.encoder.layers[-1].mlp[-2].out_features, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 1))

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # TODO Try with class token

        return self.head(self.vit.encoder(x)[:, 0])
    

class MAEViT(nn.Module):
    def __init__(self, vit, channels = 3, lighter_decoder: bool = False, cls_token: bool = False):
        super().__init__()
        self.encoder = vit
        im_dim = self.encoder.vit.patch_size**2 * channels
        self.patch_size = self.encoder.vit.patch_size
        self.hidden_dim = self.encoder.vit.hidden_dim
        decoder_num_layers, decoder_mlp_dim = (2, 2048 // 4) if lighter_decoder else (3, 2048)
        self.decoder_vit = VisionTransformer(image_size = self.encoder.vit.image_size, patch_size = self.encoder.vit.patch_size, num_layers = decoder_num_layers,num_heads = 16,
                                             hidden_dim = im_dim, mlp_dim = decoder_mlp_dim, dropout = 0.0, attention_dropout = 0.0, num_classes = 1000, has_class_token = False)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, im_dim))
        self.decoder_norm = nn.LayerNorm(im_dim)
        self.decoder_pred = nn.Linear(im_dim, im_dim, bias=True)

        self.cls_token = cls_token
    
    def mask(self, x, ratio: float = 0.25):
        x = x + self.encoder.vit.encoder.pos_embedding[:, 1: ]
        b, no_of_patches, seq_dim = x.shape
        uncovered_patches = math.ceil(ratio * no_of_patches)
        
        ids_shuffle = torch.argsort(torch.rand(b, no_of_patches, device=x.device), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        uncovered_ids = ids_shuffle[:, :uncovered_patches]
        uncovered_x = torch.gather(x, dim=1, index=uncovered_ids.unsqueeze(-1).repeat(1, 1, seq_dim))

        mask_ = torch.ones([b, no_of_patches], device=x.device)
        mask_[:, :uncovered_patches] = 0
        mask_ = torch.gather(mask_, dim=1, index=ids_restore)

        return uncovered_x, ids_restore, mask_, uncovered_ids
    
    def restore(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1) 
        x_ = torch.cat([x[:, 0:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        return x_

    def forward_encoder(self, x):
        out = self.encoder.vit.encoder.dropout(x)
        out = self.encoder.vit.encoder.layers(out)
        out = self.encoder.vit.encoder.ln(out)
    
        return out
    
    def forward_decoder(self, x):
        out = self.decoder_vit.encoder(x)
        out = self.decoder_norm(out)
        out = self.decoder_pred(out)

        return out

    def forward(self, x):
        x = self.encoder.vit._process_input(x)
        uncovered_x, ids_restore, mask_, _ = self.mask(x)

        latent = self.forward_encoder(uncovered_x)
        restored_x = self.restore(latent, ids_restore)

        if self.cls_token:
            b = x.shape[0]
            batch_class_token = self.encoder.vit.class_token.expand(b, -1, -1)
            restored_x = torch.cat([batch_class_token, restored_x], dim=1)

        out = self.forward_decoder(restored_x)
        
        if self.cls_token:
            out = out[:, 1:, :] # remove cls token

        return out, mask_



def gen_attention_map(vit, x, img):
    import numpy as np
    import cv2
    model = vit.vit
    model
    out = x
    out = model._process_input(out)

    out = out + model.encoder.pos_embedding
    out = model.encoder.dropout(out)
    out = model.encoder.layers[:-1](out)

    out = model.encoder.layers[-1].ln_1(out)
    attention, _ = model.encoder.layers[-1].self_attention(out, out, out, need_weights=False)

    attention = torch.mean(attention, dim=1)
    residual_att = torch.eye(attention.size(1))
    aug_att_mat = attention + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")