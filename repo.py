import numpy as np
import cv2

from plasma.training.data import Dataset, augmentations as augs
from albumentations import HorizontalFlip, OneOf, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate, Compose, CoarseDropout, RandomCrop


aug = Compose([
    augs.MinEdgeCrop(size=320, always_apply=True),
    HorizontalFlip(),
    OneOf([
        RandomGamma(),
        RandomBrightnessContrast(),
        #CLAHE(tile_grid_size=(5, 5))
    ], p=0.8),
    ShiftScaleRotate(shift_limit=0.1, rotate_limit=35, scale_limit=0.2, p=0.8, border_mode=cv2.BORDER_CONSTANT),
    RandomCrop(224, 224, always_apply=True),
    CoarseDropout(min_holes=1, max_holes=3, max_height=32, max_width=32, p=0.5),
])


class Data(Dataset):

    def __init__(self, df, image_path):
        super().__init__()

        self.df = df.copy().reset_index(drop=True)
        self.image_path = image_path

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]
        path = row[self.image_path]

        img = np.load(path)
        img = aug(image=img)["image"]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img[np.newaxis]
        rgb_img = rgb_img.transpose([2, 0, 1])

        return img, rgb_img
