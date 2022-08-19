import torch
from torch import nn, Tensor

from . import loss


class OHEMCrossEntropy(loss.LossBase):
    def __init__(
        self,
        ignore_label=255,
        weight=None,
        thresh=0.7,
        aux_weights=[1, 1],
        output_transform=lambda x: x,
    ) -> None:
        super(OHEMCrossEntropy, self).__init__(output_transform)
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.ce_loss_fn(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(
                [
                    w * self._forward(pred, labels)
                    for (pred, w) in zip(preds, self.aux_weights)
                ]
            )
        return self._forward(preds, labels)
