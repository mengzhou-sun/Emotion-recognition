import torch.nn as nn
from torchvision import models
import torch
class context_resnet(nn.Module):
    def __init__(self):
        super(context_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x
class body_resnet(nn.Module):
    def __init__(self):
        super(body_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x

class fusion_net(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_pose_features):
        super().__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_pose_features=num_pose_features
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2=nn.Linear(1024,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,26)
        self.d1  = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(52, 26)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body ,x_pose):
        # context_features = x_context.view(1, self.num_context_features)
        # body_features = x_body.view(1, self.num_body_features)

        pose_features=x_pose.view(-1,self.num_pose_features)

        fuse_features = torch.cat((x_context,x_body), 1)

        fuse_features=self.fc1(fuse_features)
        fuse_features=self.fc2(fuse_features)
        fuse_features=self.bn1(fuse_features)
        fuse_features0=self.fc3(fuse_features)

        total_feature=torch.cat((fuse_features0,pose_features),dim=1)
        fuse_out = self.relu(total_feature)

        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        return cat_out
