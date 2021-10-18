import math
import torch.nn.functional as func

from plasma.modules import *


class Conv_BN_ReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, dilation=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel, stride, padding, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(int(out_channels))

        if act:
            self.act = nn.ReLU(inplace=True)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_ratio, groups, att_ratio=1 / 16, down=False):
        super().__init__()

        bottleneck = math.ceil(in_channels * bottleneck_ratio)
        self.skip = nn.Sequential(*[
            Conv_BN_ReLU(in_channels, bottleneck, kernel=1, padding=0),
            Conv_BN_ReLU(bottleneck, bottleneck, stride=2 if down else 1, groups=groups),
            Conv_BN_ReLU(bottleneck, out_channels, kernel=1, padding=0, act=False),
            SEAttention(int(out_channels), ratio=att_ratio)
        ])

        if down or in_channels != out_channels:
            self.identity = Conv_BN_ReLU(in_channels, out_channels, kernel=1, padding=0,
                                         stride=2 if down else 1, act=False)
        else:
            self.identity = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        identity = self.identity(x)

        skip = self.skip(x)

        total = identity + skip
        total = self.act(total)

        return total


class Stage(nn.Sequential):

    def __init__(self, width_in, widths, bottleneck_ratio, groups, att_ratio):
        super().__init__()

        for i, width in enumerate(widths):
            self.add_module(f"block_{i}", ResBlock(width_in, width, bottleneck_ratio, groups, att_ratio, down=i == 0))
            width_in = width


class ChexNext(nn.Sequential):

    def __init__(self, stem_width, stages_width, bottleneck_ratio, groups, att_ratio, stem_kernel=3, **kwargs):
        super().__init__()

        self.stem = nn.Sequential(*[
            Normalization(),
            Conv_BN_ReLU(1, stem_width, kernel=stem_kernel, padding=3 if stem_kernel == 7 else 1, stride=2),
        ])

        width = stem_width
        for i, widths in enumerate(stages_width):
            self.add_module(f"stage_{i}",
                            Stage(width, widths, bottleneck_ratio, groups, att_ratio))
            print(widths)
            width = widths[-1]


class ChexSeg(nn.Module):

    def __init__(self, backbone, config, up_filters=(512, 256, 128, 64), n_class=1):
        super().__init__()
        widths = [w[0] for w in config["stages_width"]]
        self.stem = backbone.stem

        self.down = nn.ModuleList([getattr(backbone, f"stage_{i}") for i in range(4)])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.f3t2 = Conv_BN_ReLU(widths[3], up_filters[0])
        self.f2 = nn.Sequential(*[
            Conv_BN_ReLU(widths[2] + up_filters[0], up_filters[0]),
            ResBlock(up_filters[0], up_filters[0], 0.5, 1),
        ])

        self.f2t1 = Conv_BN_ReLU(up_filters[0], up_filters[1])
        self.f1 = nn.Sequential(*[
            Conv_BN_ReLU(widths[1] + up_filters[1], up_filters[1]),
            ResBlock(up_filters[1], up_filters[1], 0.5, 1),
        ])

        self.f1t0 = Conv_BN_ReLU(up_filters[1], up_filters[2])
        self.f0 = nn.Sequential(*[
            Conv_BN_ReLU(widths[0] + up_filters[2], up_filters[2]),
            ResBlock(up_filters[2], up_filters[2], 0.5, 1),
        ])

        self.f0ts = Conv_BN_ReLU(up_filters[2], up_filters[3])
        self.up_stem = nn.Sequential(*[
            Conv_BN_ReLU(config.stem_width + up_filters[3], up_filters[3]),
            ResBlock(up_filters[3], up_filters[3], 0.5, 1),
            Conv_BN_ReLU(up_filters[3], up_filters[3]),
            nn.Conv2d(up_filters[3], n_class, kernel_size=1),
            nn.Upsample(scale_factor=2),
        ])

    def forward(self, x):
        x = self.stem(x)

        downs = [x]
        for d in self.down:
            x = d(x)
            downs.append(x)

        f3t2 = self.f3t2(self.up(x))
        f2 = torch.cat([f3t2, downs[3]], dim=1)
        f2 = self.f2(f2)

        f2t1 = self.f2t1(self.up(f2))
        f1 = torch.cat([f2t1, downs[2]], dim=1)
        f1 = self.f1(f1)

        f1t0 = self.f1t0(self.up(f1))
        f0 = torch.cat([f1t0, downs[1]], dim=1)
        f0 = self.f0(f0)

        f0ts = self.f0ts(self.up(f0))
        up_stem = torch.cat([f0ts, downs[0]], dim=1)
        up_stem = self.up_stem(up_stem)

        return up_stem
