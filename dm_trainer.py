from typing import Tuple

import torch
import torch.nn as nn

from plasma.training import trainers
from plasma.training.trainers.utils import get_batch_tensors


class TransferTrainer(trainers.BaseTrainer):

    def __init__(self, student_model, projector, teacher_model, optimizer,
                 student_device=None, teacher_device=None):
        loss = nn.MSELoss()
        super().__init__([student_model, projector], [optimizer], loss)

        self.teacher = teacher_model
        self.student_device = student_device
        self.teacher_device = teacher_device

    def extract_data(self, batch_data):
        dtype = torch.float

        return get_batch_tensors(batch_data, dtype, self.student_device, dtype, self.teacher_device)

    def train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        student_features, student_attentions = self.models[0](inputs)
        teacher_features, teacher_attentions = self.teacher(targets)
        
        student_proj, student_cyc, teach_proj, teacher_cyc = self.models[1](student_features, teacher_features)

        if teacher_features.device != student_proj.device:
            teacher_features = teacher_features.to(student_features.device)

        loss1 = self.loss(student_proj, teacher_features)
        loss2 = self.loss(student_cyc, student_features)

        loss3 = self.loss(teach_proj, student_features)
        loss4 = self.loss(teacher_cyc, teacher_features)

        loss = loss1 + loss2 + loss3 + loss4

        att_loss = 0
        for sa, ta in zip(student_attentions, teacher_attentions):
            stage_loss = (sa - ta).pow(2) * ((sa > ta) | (ta > 0)).float()
            att_loss = att_loss + stage_loss.mean()

        loss = loss + att_loss
        loss.backward()

        d = {
            f"student_proj_loss": float(loss1),
            f"teacher_proj_loss": float(loss2),
            f"student_cycle_loss": float(loss3),
            f"teacher_cycle_loss": float(loss4),
        }
        self.optimizers[0].step()

        return {"loss": float(loss), **d}, None

    def get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        return loss_dict

    def get_eval_cache(self, inputs, targets):
        pass

    def get_eval_logs(self, eval_caches) -> dict:
        pass
