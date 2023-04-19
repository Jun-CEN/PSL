import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class TextEmbeddingHead(BaseHead):
    def __init__(self, cfg):
        super(TextEmbeddingHead, self).__init__(cfg)

    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.fc1 = nn.Linear(
            self.cfg.TEXT.WORD_EMBEDDING_DIM,
            self.cfg.TEXT.HEAD.MID_DIM,
        )

        self.fc2 = nn.Linear(
            self.cfg.TEXT.HEAD.MID_DIM,
            self.cfg.TEXT.HEAD.OUT_DIM
        )

        self.relu = nn.ReLU(inplace=True)

        if self.cfg.TEXT.HEAD.MID_LN:
            self.ln = nn.LayerNorm(self.cfg.TEXT.HEAD.MID_DIM)
    
    def forward(self, x):
        x = self.fc1(x)
        if self.cfg.TEXT.HEAD.MID_LN:
            x = self.ln(x)
        x = self.relu(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc2(x)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x