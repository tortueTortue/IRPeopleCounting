import torch
from torch import Tensor, nn
from torch.nn import MSELoss

def patchify(patch_size, imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape = (imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape = (imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def forward_loss(self, imgs, pred, mask):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = self.patchify(imgs)
    if self.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

class MaskedAutoEncoderLoss(MSELoss):
    def __init__(self, patch_size: int = 16, size_average=None, reduce=None, reduction: str = 'mean', temp: int = 5, norm_pix_loss: bool = False, is_fc: bool = False):
        super().__init__(size_average, reduce, reduction)
        self.soft = nn.Softmax(dim=1)
        self.temp = temp
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.fc = is_fc

    def forward_v2(self, reconstructed_img, img) -> Tensor:
        return super().forward(reconstructed_img, img)

    def forward(self, output, imgs) -> Tensor:
        """
                      imgs: [N, 3, H, W]
        reconstructed_imgs: [N, L, p*p*3]
                      mask: [N, L], 0 is keep, 1 is remove, 
        """
        reconstructed_imgs, mask = output
        if len(reconstructed_imgs.shape) == 4 and self.fc == True:
            b, c, _, _ = reconstructed_imgs.shape
            reconstructed_imgs = reconstructed_imgs.reshape(b, c, -1)
            reconstructed_imgs = torch.einsum('ncl->nlc', reconstructed_imgs)
        target = patchify(self.patch_size, imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (reconstructed_imgs - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
class FCMaskedAutoEncoderLoss(MaskedAutoEncoderLoss):
    def __init__(self, patch_size: int = 32, size_average=None, reduce=None, reduction: str = 'mean', temp: int = 5, norm_pix_loss: bool = False,):
        super().__init__(patch_size, size_average, reduce, reduction, temp, norm_pix_loss, is_fc =  True)