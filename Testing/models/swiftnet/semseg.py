import torch
import torch.nn as nn
from itertools import chain

from .resnet.resnet_single_scale import resnet18
from .util import _BNReluConv, upsample


class SwiftNet(nn.Module):
    def __init__(self,  num_classes=19, use_bn=False):
        super(SwiftNet, self).__init__()
        self.backbone = resnet18(use_bn=False)
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, pyramid):
        features = self.backbone(pyramid)
        logits = self.logits.forward(features[0])
        return logits

    def prepare_data(self, batch, image_size, device=torch.device('cuda')):
        if image_size is None:
            image_size = batch['target_size']
        pyramid = [p.clone().detach().requires_grad_(False).to(device) for p in batch['pyramid']]
        return {
            'pyramid': pyramid,
            'image_size': image_size,
            'target_size': batch['target_size_feats']
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        return self.forward(**data)

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
