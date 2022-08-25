import torch
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inception v2

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def ConvSigmoid(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Sigmoid()
    )






class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class Attention_Inception(nn.Module):
    def __init__(self, num_classes=10, stage='train'):
        super(Attention_Inception, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            InceptionV2ModuleA(in_channels=192,out_channels1=64, out_channels2reduce=96, out_channels2=128,
                               out_channels3reduce=16, out_channels3=32, out_channels4=32),
            InceptionV2ModuleA(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192,
                               out_channels3reduce=32, out_channels3=96, out_channels4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block4 = nn.Sequential(
            InceptionV2ModuleA(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208,
                               out_channels3reduce=16, out_channels3=48, out_channels4=64),
            InceptionV2ModuleA(in_channels=512, out_channels1=160, out_channels2reduce=112, out_channels2=224,
                               out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV2ModuleA(in_channels=512, out_channels1=128, out_channels2reduce=128, out_channels2=256,
                               out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV2ModuleA(in_channels=512, out_channels1=112, out_channels2reduce=144, out_channels2=288,
                               out_channels3reduce=32, out_channels3=64, out_channels4=64),
            InceptionV2ModuleA(in_channels=528, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                               out_channels3reduce=32, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block5 = nn.Sequential(
            InceptionV2ModuleA(in_channels=832, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                               out_channels3reduce=32, out_channels3=128, out_channels4=128),
            InceptionV2ModuleA(in_channels=832, out_channels1=384, out_channels2reduce=182, out_channels2=384,
                               out_channels3reduce=48, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block6 = nn.Sequential(
            InceptionV2ModuleA(in_channels=1024, out_channels1=512, out_channels2reduce=384, out_channels2=1024,
                               out_channels3reduce=128, out_channels3=256, out_channels4=256),
            InceptionV2ModuleA(in_channels=2048, out_channels1=512, out_channels2reduce=384, out_channels2=1024,
                               out_channels3reduce=128, out_channels3=256, out_channels4=256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(40960, num_classes)

    def forward(self, x):
        x = self.block1(x)
        # print('1', x.shape)
        x = self.block2(x)
        # print('2', x.shape)
        x = self.block3(x)
        # print('3', x.shape)
        x = self.block4(x)
        # print('4', x.shape)
        x = self.block5(x)
        # print('5', x.shape)
        x = self.block6(x)
        # print('6', x.shape)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        # print('7', x.shape)
        out = self.linear(x)
        return out

def A2_Inception(num_class):
    return Attention_Inception(num_classes=num_class)


if __name__ == '__main__':
    model = A2_Inception(num_class=81)
    input = torch.randn(2, 3, 640, 480)
    out = model(input)
    print(out.shape)