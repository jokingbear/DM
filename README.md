# DM: Diffeomorphism Matching for Fast Unsupervised Pretraining on Radiographs

This is the official implementation of the paper Diffeomorphism Matching for Fast Unsupervised Pretraining on Radiographs

To train model on 4 GPUs, run

`python train_dm.py path_to_csv`

CheXpert pretrained model is available at https://drive.google.com/u/0/uc?export=download&confirm=C4G4&id=1-3njBZ4N9VvrfoYRIFVKt9X7fEXPmzvQ

To evaluate RegChest on CheXpert valid set, use the following commands

`python evaluate.py path_to_chexpert_folder --checkpoint path_to_checkpoint`

The result should be like this

| Abnormality                | AUC      |
|----------------------------|----------|
| Enlarged Cardiomediastinum | 0.722055 |
| Cardiomegaly               | 0.833717 |
| Lung Opacity               | 0.926146 |
| Lung Lesion                | 1.000000 |
| Edema                      | 0.919929 |
| Consolidation              | 0.876828 |
| Pneumonia                  | 0.855088 |
| Atelectasis                | 0.857305 |
| Pneumothorax               | 0.971239 |
| Pleural Effusion           | 0.932523 |
| Pleural Other              | 0.995708 |
| Support Devices            | 0.952462 |
