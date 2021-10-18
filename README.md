# DM: Diffeomorphism Matching for Fast Unsupervised Pretraining on Radiographs

This is the official implementation of the paper Diffeomorphism Matching for Fast Unsupervised Pretraining on Radiographs

To train model on 4 GPUs, run

`python train_dm.py path_to_csv`

CheXpert pretrained model is available at https://drive.google.com/u/0/uc?export=download&confirm=C4G4&id=1-3njBZ4N9VvrfoYRIFVKt9X7fEXPmzvQ

To evaluate RegChest on CheXpert valid set, use the following commands

`python evaluate.py path_to_chexpert_folder path_to_checkpoint`

The result should be like this

| Abnormality                | AUC      |
|----------------------------|----------|
| Enlarged Cardiomediastinum | 0.712147 |
| Cardiomegaly               | 0.816885 |
| Lung Opacity               | 0.926440 |
| Lung Lesion                | 0.935622 |
| Edema                      | 0.923222 |
| Consolidation              | 0.872154 |
| Pneumonia                  | 0.850664 |
| Atelectasis                | 0.856250 |
| Pneumothorax               | 0.974558 |
| Pleural Effusion           | 0.924211 |
| Pleural Other              | 0.995708 |
| Support Devices            | 0.948488 |

