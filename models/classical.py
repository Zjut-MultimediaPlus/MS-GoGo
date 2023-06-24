import torch
from torchvision import models

class ALEX(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(ALEX, self).__init__()
        model_ft = models.alexnet(pretrained=pretrained)
        print(model_ft)

        '''
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft
        self.bn = torch.nn.BatchNorm1d(num_features=num_classes)
        '''

        original = list(model_ft.children())

        if not isinstance(original[-1], torch.nn.Sequential):
            modified = original[0:-1]
            self.net = torch.nn.Sequential(*modified)
            self.classifier = torch.nn.Linear(original[-1].in_features, num_classes)

        else:
            last_seq = list(original[-1].children())
            last_seq_modified = last_seq[0:-1]
            linear = torch.nn.Linear(last_seq[-1].in_features, num_classes)
            last_seq_modified.append(linear)
            # last_seq_modified.append(torch.nn.BatchNorm1d(num_features=output_dim))
            self.classifier = torch.nn.Sequential(*last_seq_modified)
            self.net = original[0:-1]
            self.net = torch.nn.Sequential(*self.net)

    def forward(self,image):
        #return self.bn(self.model(image)), None, None
        features = self.net(image)
        features = features.reshape((features.shape[0],
                                     features.shape[1] * features.shape[2] * features.shape[3]))
        output = self.classifier(features)
        return output,None,None

class VGG19(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(VGG19, self).__init__()
        model_ft = models.vgg19(pretrained=pretrained)
        model_ft.features[0] = torch.nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None

class VGG16(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(VGG16, self).__init__()
        model_ft = models.vgg19(pretrained=pretrained)
        model_ft.features[0] = torch.nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None

class RES_152(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(RES_152, self).__init__()
        model_ft = models.resnet152(pretrained=pretrained)
        model_ft.conv1 = torch.nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None

class RES_101(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(RES_101, self).__init__()
        model_ft = models.resnet101(pretrained=pretrained)
        model_ft.conv1 = torch.nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None

class RES_50(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(RES_50, self).__init__()
        model_ft = models.resnet50(pretrained=pretrained)
        model_ft.conv1 = torch.nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        print(model_ft)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None

class RES_18(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(RES_18, self).__init__()
        model_ft = models.resnet101(pretrained=pretrained)
        model_ft.conv1 = torch.nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        print(model_ft)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None

class EFFB7(torch.nn.Module):
    def __init__(self,num_classes,pretrained):
        super(EFFB7, self).__init__()
        model_ft = models.efficientnet_b7(pretrained = pretrained)
        print(model_ft)

        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self,image):
        return self.model(image), None, None
