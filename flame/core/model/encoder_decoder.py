import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional



class EncoderDecoder(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 backbone_pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.auxiliary_head = auxiliary_head

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.load_backbone_pretrained(pretrained=backbone_pretrained)

    def load_backbone_pretrained(self, pretrained: Optional[str] = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, x):
        features = self.extract_feat(x)
        output = self.decode_head(features)   # 4x reduction in image size
        output = F.interpolate(
            output, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        return output
    

class Model(nn.Module):
    def __init__(
        self,
        backbone,
        decode_head,
        pretrained: Optional[str] = None,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        backbone_pretrained=None
    ):
        super(Model, self).__init__()
        self.model = EncoderDecoder(
            backbone,
            decode_head,
            neck,
            auxiliary_head,
            train_cfg,
            test_cfg,
            backbone_pretrained
        )

        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            state_dict['state_dict'].pop("decode_head.conv_seg.weight")
            state_dict['state_dict'].pop("decode_head.conv_seg.bias")
            self.model.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
