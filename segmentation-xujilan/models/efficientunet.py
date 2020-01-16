from .EfficientUnet_Pytorch.efficientunet import *

def get_model():
    net = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True)
    return net


print(get_model())
