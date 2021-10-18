import json
import os
import torch
import numpy as np
import cv2
import pandas as pd
import argparse

from pathlib import Path
from models import regchest as classifier
from plasma.training import utils
from plasma.training.data.augmentations import MinEdgeCrop, MinEdgeResize
from plasma.modules import tta
from albumentations import Compose
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='DM Inference on CheXpert dataset')

parser.add_argument('dir', type=Path, metavar='DIR',
                    help='path to CheXpert folder')

parser.add_argument('--checkpoint', type=Path, metavar='DIR',
                    help='path to RegChest checkpoint')

parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='inference batch size')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')



def read_image(path, preprocess):
    if ".npy" in path:
        img = np.load(path)
    else:
        img = cv2.imread(path, 0)

    if img is None:
        print(path)
        
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = preprocess(image=img)["image"]
      
    return img[np.newaxis]


if __name__ == "__main__":
    args = parser.parse_args()
    
    # load model
    checkpoint = torch.load(args.checkpoint)
    model = classifier()
    print(model.load_state_dict(checkpoint))
    
    augs = [
        tta.HorizontalFlip(),
        tta.Zoom()
    ]
    
    model = tta.Compose(model, augs).cuda()
    
    # load columns
    columns = np.load('configs/columns.npy')
    
    # load csv
    df = pd.read_csv('configs/valid_groundtruth.csv')
    
    # init image preprocessing
    preprocess = Compose([
        MinEdgeResize(512, always_apply=True),
        MinEdgeCrop(positions=["center"], always_apply=True)
    ])
    
    # get data loader
    loader = utils.get_loader(str(args.dir) + '/' + df['Path'], mapper=read_image, 
                              batch_size=args.batch_size, workers=args.workers, 
                              preprocess=preprocess)
    
    # inference
    preds = []
    with utils.eval_modules(model):
        for imgs in utils.get_progress(loader, total=len(loader)):
            imgs = imgs.cuda().float()
            probs = model(imgs).mean(dim=1)
            preds.append(probs.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    
    preds = pd.DataFrame(preds, columns=columns, index=df['Path'])
    
    preds = preds[columns[:-1]]
    trues = df[columns[:-1]]
    
    aucs = {}
    for c in columns[:-1]:
        if trues[c].max() == 0:
            continue
        
        auc = roc_auc_score(trues[c], preds[c])
        aucs[c] = auc
    
    aucs = pd.Series(aucs)
    print(aucs)
    print('Mean', aucs.mean())
