"""Shared SFP/SC landmark network; state-dict layout stays checkpoint compatible."""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LandmarkHeatmapNet(nn.Module):
    def __init__(self, num_landmarks: int, input_channels: int = 3):
        super().__init__()
        self.stem = ConvBlock(input_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 192))
        self.up2 = nn.ConvTranspose2d(192, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.heatmap_head = nn.Conv2d(128, num_landmarks, 1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, num_landmarks),
        )

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        y = self.up2(x3)
        y = self.dec2(torch.cat([y, x2], dim=1))
        return self.heatmap_head(y), self.visibility_head(x3)


def heatmap_argmax(heatmaps: torch.Tensor, refine: bool = True) -> torch.Tensor:
    """Peak decoding with a bounded local log-quadratic subpixel correction.

    The compatibility name is retained for old evaluation callers. A flat or
    boundary peak keeps its integer coordinate; distant modes are not averaged.
    """
    B,C,H,W=heatmaps.shape
    flat_idx=heatmaps.reshape(B,C,H*W).argmax(dim=-1)
    y=flat_idx//W;x=flat_idx%W
    if not refine:
        return torch.stack([x.float(),y.float()],dim=-1)
    logp=heatmaps.clamp_min(1e-8).log().reshape(B*C,H,W)
    batch=torch.arange(B*C,device=heatmaps.device)
    xf=x.reshape(-1);yf=y.reshape(-1)
    center=logp[batch,yf,xf]
    left=logp[batch,yf,(xf-1).clamp(0,W-1)]
    right=logp[batch,yf,(xf+1).clamp(0,W-1)]
    top=logp[batch,(yf-1).clamp(0,H-1),xf]
    bottom=logp[batch,(yf+1).clamp(0,H-1),xf]
    def shift(a,b,valid):
        denominator=a-2*center+b
        ok=valid&(denominator < -1e-6)
        safe=torch.where(ok,denominator,torch.ones_like(denominator))
        return torch.where(ok,(.5*(a-b)/safe).clamp(-.5,.5),torch.zeros_like(center))
    dx=shift(left,right,(xf>0)&(xf<W-1)).reshape(B,C)
    dy=shift(top,bottom,(yf>0)&(yf<H-1)).reshape(B,C)
    return torch.stack([x.float()+dx,y.float()+dy],dim=-1)
