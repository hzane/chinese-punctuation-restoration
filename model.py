from torch import nn
from transformers import BertForTokenClassification


class BertPunc(nn.Module):
    def __init__(
        self,
        num_labels,
        pretrained: str = './models/hfl/chinese-macbert-base',
    ):
        super(BertPunc, self).__init__()
        self.model = BertForTokenClassification.from_pretrained(
            pretrained,
            num_labels = num_labels,
        )

    def forward(self, x):
        x = self.model(x)[0]
        return x
