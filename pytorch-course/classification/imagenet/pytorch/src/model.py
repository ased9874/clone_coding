from torch import nn, Tensor
from torchvision.models import (  #사전 정의된 모델 아키텍처 함수와 
    vgg16, VGG16_Weights,         #각각의 모델에 대한 사전학습된 가중치(weight) 종류를 정의한 클래스
    resnet18, ResNet18_Weights,   
    efficientnet_b0, EfficientNet_B0_Weights
)
# 예 model = create_model('resnet')  # → ResNet18 모델이 불러와짐
def create_model(model: str):  
    if model == 'vgg':
        return _vgg16_pretrained()
    elif model == 'resnet':
        return _resnet18_pretrained()
    elif model == 'efficientnet':
        return _efficientb0_pretrained()
    
def _vgg16_pretrained():      # ImageNet으로 사전학습된 가중치를 넣어 VGG-16으로 변환. 이하 동일
    return vgg16(weights = (VGG16_Weights.IMAGENET1K_V1))

def _resnet18_pretrained(): 
    return resnet18(weights = (ResNet18_Weights.IMAGENET1K_V1))

def _efficientb0_pretrained():
    return efficientnet_b0(weights = (EfficientNet_B0_Weights.IMAGENET1K_V1))