import torchvision 
import torch.nn as nn 
import torch 
class Resnet18(nn.Module):
    def __init__(self,out_features):
        super(Resnet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = PredictionHead(self.model.fc.in_features,out_features)

    def forward(self,x):
        return self.model(x)

class Resnet34(nn.Module):
    def __init__(self,out_features):
        super(Resnet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=False)
        self.model.fc = PredictionHead(self.model.fc.in_features,out_features)

    def forward(self,x):
        return self.model(x)

class PredictionHead(nn.Module):
    def __init__(self,in_features,out_features):
        super(PredictionHead,self).__init__()
        self.fc1 = nn.Linear(in_features,out_features)
        self.fc2 = nn.Linear(in_features,out_features)
        self.fc3 = nn.Linear(out_features*2,1)
    def forward(self,x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        norm1 = torch.linalg.norm(y1, dim=1)
        norm2 = torch.linalg.norm(y2,dim=1)
        y3 = torch.nn.functional.sigmoid(self.fc3(torch.cat([y1,y2],dim=1)).reshape(-1))
        return y1, y2, y3