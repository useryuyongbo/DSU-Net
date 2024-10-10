import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                        groups=in_channels)

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        result = x * out

        return result


class NIF(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()

        )
        self.sigmoid = nn.Sigmoid()
        self.sSE = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels, 1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, stride=2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, dilation=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        
        x1 = self.sSE(x1)
        x_2 = self.conv(torch.cat([x1, x2], dim=1))
        x_2 = x_2 * self.cSE(x_2)
        x = x_2 + x2
        
        return x

class DCM(nn.Module):
    def __init__(self, in_channels):
        super(DCM, self).__init__()
        self.cbam = CBAM(in_channels)
        self.se = SEBlock(in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, padding=3, dilation=3, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.cbam(x)
        x2 = self.se(x)
        out = self.conv1(torch.cat([x1, x2], dim=1))
        out = out + x1
        result = self.conv2(out)

        return result

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

class DConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DConv, self).__init__(

            DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, channel, bilinear=True):
        super(Up, self).__init__()
        if channel == 1:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)
        else:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DConv(in_channels, out_channels)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DConv(in_channels, out_channels)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class ConvR(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvR, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

class FRM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FRM, self).__init__()
        self.conv1 = ConvR(in_channels, in_channels // 2)
        self.conv2 = ConvR(in_channels // 2, in_channels // 4)
        self.conv3 = ConvR(in_channels // 4, in_channels // 2)
        self.conv4 = ConvR(in_channels // 2, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.out = OutConv(in_channels, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = x3 + x1
        x4 = self.conv4(x3)
        x4 = x4 + x
        out = self.out(x4)
        return out
    
class DSU_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(DSU_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        # self.up1 = Up(base_c * 16, base_c * 8 // factor, 1, bilinear)
        # self.up2 = Up(base_c * 8, base_c * 4 // factor, 1, bilinear)
        # self.up3 = Up(base_c * 4, base_c * 2 // factor, 1, bilinear)
        # self.up4 = Up(base_c * 2, base_c, 1, bilinear)

        self.up21 = Up(base_c * 16, base_c * 8 // factor, 2, bilinear)
        self.up22 = Up(base_c * 8, base_c * 4 // factor, 2, bilinear)
        self.up23 = Up(base_c * 4, base_c * 2 // factor, 2, bilinear)
        self.up24 = Up(base_c * 2, base_c, 2, bilinear)

        self.frm = FRM(base_c, num_classes)

        self.dcm = DCM(base_c * 8)

        self.se1 = SEBlock(base_c)
        self.nif1 = NIF(base_c * 2)
        self.nif2 = NIF(base_c * 4)
        self.nif3 = NIF(base_c * 8)


    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xc_1 = self.se1(x1)
        xc_2 = self.nif1(x1, x2)
        xc_3 = self.nif2(x2, x3)
        xc_4 = self.nif3(x3, x4)

        x5 = self.dcm(x5)

        # x_11 = self.up1(x5, x4)
        x_21 = self.up21(x5, xc_4)

        # x_12 = self.up2(x_11, x3)
        x_22 = self.up22(x_21, xc_3)

        # x_13 = self.up3(x_12, x2)
        x_23 = self.up23(x_22, xc_2)
        
        # x_14 = self.up4(x_13, x1)
        x_24 = self.up24(x_23, xc_1)
        
        out = self.frm(x_24)

        return out
