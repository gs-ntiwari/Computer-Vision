import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1= nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2= nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #self.conv3 = nn.Conv2d(32, 16, 2)
        #self.pool3 = nn.MaxPool2d(2, 2)
        #self.bn3= nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(64*6*6, 600)
        self.fc2 = nn.Linear(600, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x=self.bn1(x)
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x=self.bn2(x)
        #print(x.shape)
        #x = self.pool3(F.relu(self.conv3(x)))
        #x=self.bn3(x)
        x = x.view(-1, 64*6*6)
        nn.Dropout(0.5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x= F.log_softmax(self.fc3(x))
        return x
