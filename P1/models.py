## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.convstart_bn = nn.BatchNorm2d(1)
        
        self.conva = nn.Conv2d(1, 16, 5) #220
        self.conva_bn = nn.BatchNorm2d(16)
        self.maxpoola = nn.MaxPool2d(2,2) #110
        #self.dropa = nn.Dropout2d(p=0.1)
        
        self.conv1 = nn.Conv2d(16, 32, 4) #107
        self.conv1_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2,2) #53
        #self.drop1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3) #51
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2,2) #25
        #self.drop2 = nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 2) #24
        self.conv3_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2,2) #12
        #self.drop3 = nn.Dropout2d(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1) #12
        self.conv4_bn = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(2,2) #6
        #self.drop4 = nn.Dropout2d(p=0.4)

        self.fc1 = nn.Linear(256*6*6, 1000)
        self.drop5 = nn.Dropout2d(p=0.2)

        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout2d(p=0.3)

        self.fc3 = nn.Linear(1000, 136)


        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting



    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        x = self.convstart_bn(x)
        x = self.maxpoola(F.relu(self.conva_bn(self.conva(x))))
        x = self.maxpool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.maxpool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.maxpool3(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.maxpool4(F.relu(self.conv4_bn(self.conv4(x))))
        x = x.view(x.size(0), -1)

        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x