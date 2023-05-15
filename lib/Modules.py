# -*- coding: utf-8 -*-
import torch.nn as nn

affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from lib.GatedConv import GatedConv2dWithActivation



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


"""
    Position Attention Module (PAM)
"""


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


"""
    ASPP
"""

class GPM(nn.Module):
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=128):
        # def __init__(self, dilation_series=[2, 5, 7], padding_series=[2, 5, 7], depth=128):
        super(GPM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(2048, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(2048, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth * 5, 256, kernel_size=3, padding=1),
            PAM(256)
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=affine_par),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out


"""
    R-Net
"""


class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


"""
    Joint Attention module (CA + SA)
"""

class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

class CA(nn.Module):
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class AM(nn.Module):
    def __init__(self, channels, lf):
        super(AM, self).__init__()
        self.CA = CA(lf=lf)
        self.SA = SA(channels)

    def forward(self, x):
        x = self.CA(x)
        x = self.SA(x)
        return x


"""
    Low-Frequency Attention Module (LFA)
"""


class RB(nn.Module):
    def __init__(self, channels, lf):
        super(RB, self).__init__()
        self.RB = BasicConv2d(channels, channels, 3, padding=1, bn=nn.InstanceNorm2d if lf else nn.BatchNorm2d)

    def forward(self, x):
        y = self.RB(x)
        return y + x


class ARB(nn.Module):
    def __init__(self, channels, lf):
        super(ARB, self).__init__()
        self.lf = lf
        self.AM = AM(channels, lf)
        self.RB = RB(channels, lf)

        self.mean_conv1 = ConvLayer(1, 16, 1, 1)
        self.mean_conv2 = ConvLayer(16, 16, 3, 1)
        self.mean_conv3 = ConvLayer(16, 1, 1, 1)

        self.std_conv1 = ConvLayer(1, 16, 1, 1)
        self.std_conv2 = ConvLayer(16, 16, 3, 1)
        self.std_conv3 = ConvLayer(16, 1, 1, 1)

    def PONO(self, x, epsilon=1e-5):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
        output = (x - mean) / std
        return output, mean, std

    def forward(self, x):
        if self.lf:
            x, mean, std = self.PONO(x)
            mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
            std = self.std_conv3(self.std_conv2(self.std_conv1(std)))
        y = self.RB(x)
        y = self.AM(y)
        if self.lf:
            return y * std + mean
        return y


"""
    Guidance-based Upsampling
"""

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


"""
    Guidance-based Feature Aggregation Module (GFA)
"""


class AGF(nn.Module):
    def __init__(self, channels, lf):
        super(AGF, self).__init__()
        self.ARB = ARB(channels, lf)
        self.GF = GF(r=2, eps=1e-2)

    def forward(self, high_level, low_level):
        N, C, H, W = high_level.size()
        high_level_small = F.interpolate(high_level, size=(int(H / 2), int(W / 2)), mode='bilinear', align_corners=True)
        y = self.ARB(low_level)
        y = self.GF(high_level_small, low_level, high_level, y)
        return y


class AGFG(nn.Module):
    def __init__(self, channels, lf):
        super(AGFG, self).__init__()
        self.GF1 = AGF(channels, lf)
        self.GF2 = AGF(channels, lf)
        self.GF3 = AGF(channels, lf)

    def forward(self, f1, f2, f3, f4):
        y = self.GF1(f2, f1)
        y = self.GF2(f3, y)
        y = self.GF3(f4, y)
        return y



"""
    Deep Wavelet-like Decomposition (DWD): LWD + HFA/LFA + GFA
"""


class GCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM3, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)

        # wavelet attention module
        self.DWT = DWT()
        self.AGFG_LL = AGFG(out_channels, True)
        self.AGFG_LH = AGFG(out_channels, False)
        self.AGFG_HL = AGFG(out_channels, False)
        self.AGFG_HH = AGFG(out_channels, False)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)

        wf1 = self.DWT(f1)
        wf2 = self.DWT(f2)
        wf3 = self.DWT(f3)
        wf4 = self.DWT(f4)

        LL = self.AGFG_LL(wf4[0], wf3[0], wf2[0], wf1[0])
        LH = self.AGFG_LH(wf4[1], wf3[1], wf2[1], wf1[1])
        HL = self.AGFG_HL(wf4[2], wf3[2], wf2[2], wf1[2])
        HH = self.AGFG_HH(wf4[3], wf3[3], wf2[3], wf1[3])
        return LL, LH, HL, HH, f1, f2, f3, f4



"""
    TFD
"""


class TFD(nn.Module):
    def __init__(self, in_channels):
        super(TFD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.gatedconv = GatedConv2dWithActivation(in_channels * 2, in_channels, kernel_size=3, stride=1,
                                                   padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
                                                   activation=torch.nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feature_map, perior_repeat):
        assert (feature_map.shape == perior_repeat.shape), "feature_map and prior_repeat have different shape"
        uj = perior_repeat
        uj_conv = self.conv(uj)
        uj_1 = uj_conv + uj
        uj_i_feature = torch.cat([uj_1, feature_map], 1)
        uj_2 = uj_1 + self.gatedconv(uj_i_feature) - 3 * uj_conv
        return uj_2


"""
    Ordinary Differential Equation (ODE)
"""


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        alpha = self.getalpha(torch.cat([f1, f2], dim=1))
        out = feature_map + f1 * alpha + f2 * (1 - alpha)
        return out

"""
    segmentation-oriented edge-assited decoder (SED)
"""

class REU6(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(REU6, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.TFD = TFD(in_channels)
        self.out_B = nn.Sequential(
            BasicDeConv2d(in_channels, mid_channels // 2, kernel_size=3, stride=2, padding=1, out_padding=1),
            BasicConv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.ode = ODE(in_channels)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=x.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size()[1], -1, -1)], dim=1))

        ode_out = self.ode(yt)
        bound = self.out_B(ode_out)
        bound = self.edge_enhance(bound)

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y = r_prior_cam.expand(-1, x.size()[1], -1, -1).mul(x)

        cat2 = torch.cat([y, ode_out], dim=1)  # 2,128,48,48

        y = self.out_y(cat2)
        y = y + prior_cam
        return y, bound

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class REM11(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(REM11, self).__init__()
        self.REU_f1 = REU6(in_channels, mid_channels)
        self.REU_f2 = REU6(in_channels, mid_channels)
        self.REU_f3 = REU6(in_channels, mid_channels)
        self.REU_f4 = REU6(in_channels, mid_channels)

    def forward(self, x, prior_0, pic):
        f1, f2, f3, f4 = x

        f4_out, bound_f4 = self.REU_f4(f4, prior_0)  # b,1,12,12 b,1,48,48
        f4 = F.interpolate(f4_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f4 = F.interpolate(bound_f4, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)  # b,1,24,24 b,1,96,96
        f3 = F.interpolate(f3_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f3 = F.interpolate(bound_f3, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)  # b,1,48,48 b,1,192,192
        f2 = F.interpolate(f2_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f2 = F.interpolate(bound_f2, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)  # b,1,96,96 b,1,384,384
        f1 = F.interpolate(f1_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f1 = F.interpolate(bound_f1, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        return f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


if __name__ == '__main__':
    f1 = torch.randn(2, 1, 12, 12).cuda()
    ll = torch.randn(2, 64, 96, 96).cuda()
    lh = torch.randn(2, 64, 48, 48).cuda()
    hl = torch.randn(2, 64, 24, 24).cuda()
    hh = torch.randn(2, 64, 12, 12).cuda()
    pict = torch.randn(2, 3, 384, 384).cuda()
    x = [ll, lh, hl, hh]
    rem = REM11(64, 64).cuda()
    f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = rem(x, f1, pict)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)
