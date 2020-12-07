import json
import torchvision


if __name__ == '__main__':
    print(torchvision.models.vgg16(pretrained=True).features)