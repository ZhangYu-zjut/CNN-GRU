# encoding:utf-8
# This is the CNNNGRU model
# encoding:utf-8
import torch
from torch.autograd import Variable as V
import torchvision
import torch.nn.functional as F

import torchvision
from torchvision.models import resnet18
#from myutils import Flatten
import torch
import torch.nn as nn

from config import parsers
opt = parsers()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
        
class CNNGRU(nn.Module):
    def __init__(self,cnn_output_dim=1,gru_seq_length=opt.window,gru_input_dim=1,gru_hidden_dim=10,
                 gru_batch_size=opt.batch_size,gru_output_dim=1,gru_layer_num=2):
        super(CNNGRU, self).__init__()
        self.cnn = CNN(output_dim=cnn_output_dim)
        self.gru = GRU(gru_seq_length,gru_input_dim,gru_hidden_dim,gru_batch_size,gru_output_dim,gru_layer_num)
        self.fc = nn.Linear((cnn_output_dim+gru_output_dim), 1)

    def forward(self,img_data,irr_data):
        """
        irr_data shape:
        [seq_length,batch_size,input_dim]->[opt.window,opt.batchsize,1]->[5,16,1]
        """
        #torch.backends.cudnn.benchmark = True
        y1 = self.cnn(img_data)
        irr_data = irr_data.to(torch.float32)
        y2 = self.gru(irr_data)
        combine = torch.cat((y1,y2),1)
        out = self.fc(combine)
        return (out)

class CNN(nn.Module):
    def __init__(self,output_dim):
        super(CNN,self).__init__()
        trained_model = resnet18(pretrained=True)
        self.layer = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                          Flatten(), # [b, 512, 1, 1] => [b, 512]
                          nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(512, output_dim),
                          
                          )
        pass

    def forward(self,x):
        x = self.layer(x)
        return x
        pass

class GRU(nn.Module):

    def __init__(self,seq_length,input_dim,hidden_dim,batch_size,output_dim=1,layer_num=2):
        super(GRU,self).__init__()
        self.seq_legnth = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.gru = nn.GRU(self.input_dim,self.hidden_dim,self.layer_num)
        self.fc = nn.Linear(self.hidden_dim,self.output_dim)
        pass

    def init_hidden(self):
        return (torch.zeros(self.layer_num,self.batch_size,self.hidden_dim).to(device))
    """
    input0 shape:[seq_length,batch_size,input_dim]
    input1 shape:[num_layers,batch_size,hidden_dim]
    """
    def forward(self,x):
        # [batch_size,seq_length]->[input_dim=1,batch_size,seq_length]->[seq_length,batch_size,input_dim]
        x = torch.unsqueeze(x,0)
        x = x.permute(2,1,0)
        self.batch_size = x.size(1)
        self.hidden = self.init_hidden()
        gru_output,self.hidden = self.gru(x,self.hidden)
        y_pred = self.fc(gru_output[-1])
        return y_pred
        pass


if __name__ == '__main__':
    # Module test
    seq_length = 7
    input_dim = 20
    batch_size = 6
    hidden_dim = 20
    model = GRU(input_dim,hidden_dim,batch_size)

    inpus = torch.randn(size=(seq_length,batch_size,input_dim)).to(device)
    model = model.to(device)
    output = model(inpus)
    print(output)
    print(output.size()) # [batch_size, output_dim]
