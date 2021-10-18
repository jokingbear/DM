import torch
import torch.nn as nn
import torch.optim as opts

import pandas as pd
import numpy as np
import os
import argparse

import repo
import models

from diff_transfer_trainer_2 import TransferTrainer as Trainer
from plasma.training import utils, callbacks

parser = argparse.ArgumentParser(description='train DM')


parser.add_argument('csv', type=Path, metavar='DIR',
                    help='path to csv file containing all path to radiographs')

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='DIR',
                    help='where to save checkpoint')

parser.add_argument('--batch_size', default=128, type=int, metavar='DIR',
                    help='mini batch size')

parser.add_argument('--workers', default=32, type=int, metavar='DIR',
                    help='numbers of worker')



if __name__ == "__main__":
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    train = repo.Data(df, "path")
    print('dataset length', len(train))
    
    teacher = teacher = models.Teacher()
    teacher = nn.DataParallel(teacher).cuda()
    
    student = models.Student()
    student = nn.DataParallel(student).cuda()
    
    projector = models.CycleProjector()
    projector = nn.DataParallel(projector).cuda()
    
    opt = opts.SGD([*student.parameters(), *projector.parameters()], lr=1.5e-1, momentum=0.9, nesterov=True, weight_decay=1e-6)
    trainer = Trainer(student, projector, teacher, opt, "cuda:0", "cuda:0")
    
    cbs = [
        callbacks.SuperConvergence(epochs=8, directory=args.checkpoint, name='regchest.pth'),
        callbacks.CSVLogger(f"regchest.csv", append=True),
    ]
    
    loader = train.get_torch_loader(batch_size=args.batch_size, workers=args.workers)
    trainer.fit(loader, callbacks=cbs)
