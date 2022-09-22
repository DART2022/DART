# coding: UTF-8
import torch
import torch.nn as nn

from opts.opts import INFO
from .ssim import ssim
from .percep_loss import VGG19

"""
    @date:   2020.05.06
    @author: samuel ko
    @target: transport the loss of original keras repo to pytorch.
"""


def get_loss_from_name(name):
    if name == "l1":
        return L1LossWrapper()
    elif name == 'l2':
        return L2LossWrapper()


class TotalLoss(nn.Module):
    def __init__(self,
                 apply_grad_pen=False,
                 grad_pen_weight=None,
                 entropy_qz=None,
                 regularization_loss=None,
                 beta=1e-4,
                 loss='l2'):
        super(TotalLoss, self).__init__()

        # Get the losses
        self.loss = get_loss_from_name(loss)
        self.embed_loss = EmbeddingLoss()
        self.grad_loss = GradPenLoss()

        # Extra parameters
        self.apply_grad_pen = apply_grad_pen
        self.grad_pen_weight = grad_pen_weight
        self.entropy_qz = entropy_qz
        self.regularization_loss = regularization_loss
        self.beta = beta
        # if torch.cuda.is_available():
        #     return loss.cuda()

    def forward(self, pred_img, gt_img, embedding):

        # print("预测", pred_img.shape)
        loss = self.loss(pred_img, gt_img).mean(dim=[1, 2])
        # print("损失", loss.shape)
        loss += self.beta * self.embed_loss(embedding)

        if self.apply_grad_pen:
            loss += self.grad_pen_weight * self.grad_loss(self.entropy_qz, embedding, pred_img)
        if self.entropy_qz is not None:
            loss -= self.beta * self.entropy_qz
        if self.regularization_loss is not None:
            loss += self.regularization_loss

        return loss.mean()


# Wrapper of the L1Loss so that the format matches what is expected

class L1LossWrapper(nn.Module):
    def __init__(self):
        super(L1LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean(torch.abs(pred_img - gt_img), dim=1)


class L2LossWrapper(nn.Module):
    def __init__(self):
        super(L2LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean((pred_img - gt_img) ** 2, dim=1)


class EmbeddingLoss(nn.Module):
    def forward(self, embedding):
        return (embedding ** 2).mean(dim=1)


class GradPenLoss(nn.Module):
    def forward(self, entropy_qz, embedding, y_pred):
        if entropy_qz is not None:
            return torch.mean((entropy_qz * torch.autograd.grad(y_pred ** 2,
                                                                embedding)) ** 2)  # No batch shape is there so mean accross everything is ok
        else:
            return torch.mean((torch.autograd.grad(y_pred ** 2, embedding)) ** 2)
