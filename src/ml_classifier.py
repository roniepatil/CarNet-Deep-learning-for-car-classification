import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import ml_decoder

class EfficientNet_b2(nn.Module):
    def __init__(self):
        super(EfficientNet_b2,self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        self.classifier_layer = nn.Sequential(
            # nn.Linear(1408 , 512),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.2),
            # nn.Linear(512 , 256),
            # nn.Linear(256 , 196)
            ml_decoder.MLDecoder(num_classes=196, initial_num_features=1408)
        )
    
    def forward(self,inputs):
        x = self.model.extract_features(inputs)
        # x = self.model._avg_pooling(x)
        # x = x.flatten(start_dim=1)
        # x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x