import torch.hub as hub
import math
import plasma.hub as phub
import numpy as np

from plasma.modules import *
from scipy.stats import norm
from blocks import ChexNext


# code reuse from https://github.com/clovaai/overhaul-distillation
def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.tensor(margin, dtype=torch.float).to(std.device)


def regchest(**kwargs):
    config = pd.read_json(f"configs/regchest.json", orient="index")[0]
    final_width = int(config["stages_width"][-1][0])

    features = ChexNext(**config, **kwargs)
    avg_pool = nn.Sequential(*[
        SAModule(final_width),
        GlobalAverage(),
    ])
    classifier = nn.Sequential(nn.Linear(final_width, 14), nn.Sigmoid())

    model = nn.Sequential()
    model.features = features
    model.avg_pool = avg_pool
    model.classifier = classifier

    return model


class Teacher(nn.Module):

    def __init__(self, resnext='48d'):
        super().__init__()

        features = hub.load('facebookresearch/WSL-Images', f'resnext101_32x{resnext}_wsl')

        stem = nn.Sequential(*[
            ImagenetNorm(),
            features.conv1,
            features.relu,
        ])

        relu1 = features.layer1[-1].relu
        bn1 = features.layer1[-1].bn3
        features.layer1[-1].relu = Identity()

        relu2 = features.layer2[-1].relu
        bn2 = features.layer2[-1].bn3
        features.layer2[-1].relu = Identity()

        relu3 = features.layer3[-1].relu
        bn3 = features.layer3[-1].bn3
        features.layer3[-1].relu = Identity()

        relu4 = features.layer4[-1].relu
        bn4 = features.layer4[-1].bn3
        features.layer4[-1].relu = Identity()

        margins = [get_margin_from_BN(bn) for bn in [bn1, bn2, bn3, bn4]]
        self.margins = margins

        self.stages = nn.ModuleList([
            stem,
            nn.Sequential(features.maxpool, features.layer1),
            nn.Sequential(relu1, features.layer2),
            nn.Sequential(relu2, features.layer3),
            nn.Sequential(relu3, features.layer4),
        ])

        self.global_pool = nn.Sequential(*[
            relu4,
            GlobalAverage(),
        ])

    @torch.no_grad()
    def forward(self, x):
        features = x
        attention_maps = []

        for i, stage_module in enumerate(self.stages):
            #assert stage_module.training

            features = stage_module(features)

            if i > 0:
                projection_map = torch.max(features, self.margins[i - 1][np.newaxis, :, np.newaxis, np.newaxis].to(x.device))
                attention_maps.append(projection_map)

        return self.global_pool(features), attention_maps


class Student(nn.Module):

    def __init__(self):
        super().__init__()

        features = regchest().features
        
        student_widths = [80, 160, 480, 1200, 3120]
        teacher_widths = [64, 256, 512, 1024, 2048]

        projectors = [nn.Sequential(nn.Conv2d(in_channels=sw, out_channels=tw, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(tw))
                      for sw, tw in zip(student_widths[1:], teacher_widths[1:])]

        self.stages = nn.ModuleList([
            features.stem,
            features.stage_0,
            features.stage_1,
            features.stage_2,
            features.stage_3,
        ])

        self.global_pool = GlobalAverage()

        self.projectors = nn.ModuleList(projectors)

    def forward(self, x):
        features = x
        attention_maps = []

        for i, m in enumerate(self.stages):
            features = m(features)

            if i > 0:
                project = self.projectors[i - 1](features)
                attention_maps.append(project)

        return self.global_pool(features), attention_maps


class CycleProjector(nn.Module):

    def __init__(self):
        super().__init__()

        teacher_widths = [64, 256, 512, 1024, 2048]
        student_widths = [80, 160, 480, 1200, 3120]

        self.student_projector = nn.Sequential(*[
            nn.Linear(student_widths[-1], student_widths[-1]),
            nn.BatchNorm1d(student_widths[-1]),
            nn.ELU(inplace=True),
            
            nn.Linear(student_widths[-1], student_widths[-1]),
            nn.BatchNorm1d(student_widths[-1]),
            nn.ELU(inplace=True),
            
            nn.Linear(student_widths[-1], teacher_widths[-1]),
        ])

        self.teacher_projector = nn.Sequential(*[
            nn.Linear(teacher_widths[-1], teacher_widths[-1]),
            nn.BatchNorm1d(teacher_widths[-1]),
            nn.ELU(inplace=True),
            
            nn.Linear(teacher_widths[-1], teacher_widths[-1]),
            nn.BatchNorm1d(teacher_widths[-1]),
            nn.ELU(inplace=True),
          
            nn.Linear(teacher_widths[-1], student_widths[-1]),
        ])

    def forward(self, student_feature, teacher_feature):
        student_project = self.student_projector(student_feature)
        student_cycle = self.teacher_projector(student_project)

        teacher_project = self.teacher_projector(teacher_feature)
        teacher_cycle = self.student_projector(teacher_project)
        return student_project, student_cycle, teacher_project, teacher_cycle
