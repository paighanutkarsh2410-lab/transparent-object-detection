import torch
import torch.nn as nn
import pytorch_msssim

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, use_ssim=True, device="cpu"):
        super(UncertaintyWeightedLoss, self).__init__()
        self.use_ssim = use_ssim
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = pytorch_msssim.SSIM(data_range=1.0, channel=3)
        self.log_vars = nn.Parameter(torch.zeros((2 if use_ssim else 1)), requires_grad=True).to(device)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        if self.use_ssim:
            ssim = 1 - self.ssim_loss(pred, target)  # SSIM loss
            loss = torch.exp(-self.log_vars[0]) * l1 + self.log_vars[0] + \
                   torch.exp(-self.log_vars[1]) * ssim + self.log_vars[1]
            return loss, {"l1": l1.item(), "ssim": ssim.item()}
        else:
            loss = torch.exp(-self.log_vars[0]) * l1 + self.log_vars[0]
            return loss, {"l1": l1.item()}
