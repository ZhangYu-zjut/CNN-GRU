# implement of resnet34
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision
import torch.nn.functional as F
import torchsummary
from config import parsers
opt = parsers()

class ResidualBlock(nn.Module):
    def __init__(self,in_feature,out_feature,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
         nn.Conv2d(in_feature,out_feature,kernel_size=3,stride=stride,padding=1,bias=False),
         nn.BatchNorm2d(out_feature),
         nn.ReLU(),
         #这里也是两个out
         nn.Conv2d(out_feature,out_feature,kernel_size=3,stride=1,padding=1,bias=False),
         nn.BatchNorm2d(out_feature),       
        
        )     
        self.right = shortcut 
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
        pass
    
class ResNet18(nn.Module):
    # 在init里面，定义模型内部的结构
    def __init__(self,num_classes):
        super(ResNet18,self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self._make_layer(64,128,2)
        self.layer2 = self._make_layer(128,256,2,stride=2)
        self.layer3 = self._make_layer(256,512,2,stride=2)
        self.layer4 = self._make_layer(512,512,2,stride=2)
        self.fc = nn.Linear(512,num_classes)

    def _make_layer(self,in_feature,out_feature,block_num,stride=1):
        shortcut = nn.Sequential(nn.Conv2d(in_feature,out_feature,1,stride),
                                  nn.BatchNorm2d(out_feature))
        layer_list = []
        layer_list.append(ResidualBlock(in_feature,out_feature,stride,shortcut))
        for i in range(1,block_num):
            # 这里的两个输出通道是一样的
            layer_list.append(ResidualBlock(out_feature,out_feature))
        return nn.Sequential(*layer_list)
        pass
    
    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,7) # fc之前的平均pool
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        #x = x>opt.irr_max?opt.irr_max:x
        #x = x<opt.irr_min?opt.irr_min:x
        x = F.sigmoid(x)
        return x
        pass
