import torch.nn as nn
import ml_decoder as mld
from torchvision import models
import ml_classifier
from torchinfo import summary

def build_model(pretrained=True, fine_tune=True, num_classes=196):
    Num_classes = num_classes
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    # model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    # model = models.efficientnet_b7(pretrained=pretrained)
    model = ml_classifier.EfficientNet_b2()

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    # model.classifier[1] = nn.Linear(in_features=1408, out_features=Num_classes)
    
    # model.global_pool = nn.Identity()
    # del model.fc
    # print(model.classifier)
    # del model.classifier
    # del model.avgpool
    # model.avgpool = nn.Identity()
    # removed = list(model.children())[:-1]
    # model = nn.Sequential(*self.removed)
    # print(model)
    # model.avgpool = ml_classifier.finalclassifier()
    # model.classifier = mld.MLDecoder(num_classes=Num_classes, initial_num_features=1408)
    # summary(model.cuda(),input_size=(32,3,224,224))
    # print(model)
    # model = mld.add_ml_decoder_head(model=model, num_classes=196, num_f=1408)
    return model