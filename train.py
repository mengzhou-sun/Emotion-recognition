from Datalode import EmoticDataset
from torch.utils.data import  DataLoader
import torchvision.models as models
from Model import context_resnet,body_resnet,fusion_net
from GraphConvolution import GCN1
import torch.optim as optim
import torch
import dgl
from loss import *
#from loss import DiscreteLoss
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")
dataset = EmoticDataset(mode="train", transforms1=True)
train_loader = DataLoader(dataset=dataset,batch_size=2,
                                   shuffle=True)
dataset1=EmoticDataset(mode="test",transforms1=True)
validation_loader=DataLoader(dataset=dataset1,batch_size=2,shuffle=True)


net1=context_resnet()
net2=body_resnet()
net3=GCN1(2,18,26)
fusion_net=fusion_net(2048,2048,26)

net1=net1.to(device)
net2=net2.to(device)
net3=net3.to(device)
fusion_net=fusion_net.to(device)
# for param in net1.parameters():
#         param.requires_grad = True
# for param in net2.parameters():
#         param.requires_grad = True
# for param in net3.parameters():
#         param.requires_grad = True
# for param in fusion_net.parameters():
#         param.requires_grad = True

opt = optim.Adam(
    (list(net1.parameters()) + list(net2.parameters()) + list(net3.parameters())+list(fusion_net.parameters())),
    lr=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5, eta_min=1e-5)
loss = torch.nn.BCEWithLogitsLoss()
loss1= DiscreteLoss()
for e in range(30):
    train_loss=0.0
    net1.train()
    net2.train()
    net3.train()
    fusion_net.train()
    for image_context1,image_body1,pose_array,label  in iter (train_loader):
        image_context1=image_context1.to(device)
        image_body1=image_body1.to(device)
        label=label.to(device)
        image_context = image_context1.permute(0, 3, 1, 2)
        image_body=image_body1.permute(0,3,1,2)
        opt.zero_grad()
        scheduler.step(e)
        lr = scheduler.get_lr()
        feature_body=net1(image_body)
        feature_context=net2(image_context)
        pose_array = pose_array.permute(1, 2, 0)
        pose_array = torch.squeeze(pose_array, dim=2)

        u, v = torch.tensor([[4, 3, 7, 6, 13, 12, 10, 9, 11, 8, 5, 2, 0, 15, 14, 17, 16],
                             [3, 2, 6, 5, 12, 11, 9, 8, 5, 2, 1, 1, 1, 0, 0, 15, 14]])
        g = dgl.graph((u, v))
        bg = dgl.to_bidirected(g)
        bg.ndata['feat'] = pose_array
        bg=bg.to(device)
        pose_feature=net3(bg, bg.ndata['feat'].float())
        prediction=fusion_net(feature_context,feature_body,pose_feature)
        # y_pred1 = prediction.cpu().data
        # y = label.cpu().data
        train_loss1=loss(prediction,label)
        #accuracy1=accuracy(y_pred1,y)
        train_loss += train_loss1.item()
        # train_loss.requires_grad_(True)
        train_loss1.backward()
        opt.step()
    print('epoch = %d train_loss = %.4f  ' % (e, train_loss))
    # val_loss=0.0
    # net1.eval()
    # net2.eval()
    # net3.eavl()
    # fusion_net.eval()
    # with torch.no_grad():
    #     for iter,(images_context1, images_body1, bg, labels) in enumerate(validation_loader):
    #         image_context1 = image_context1.to(device)
    #         image_body1 = image_body1.to(device)
    #         bg = bg.to(device)
    #         label = label.to(device)
    #         image_context = image_context1.permute(0, 3, 1, 2)
    #         image_body = image_body1.permute(0, 3, 1, 2)
    #         feature_body = net1(image_body)
    #         feature_context = net2(image_context)
    #         pose_feature = net3(bg, bg.ndata['feat'].float())
    #         prediction = fusion_net(feature_context, feature_body, pose_feature)
    #         val_loss = loss(prediction, labels)
    #         val_loss+=val_loss.item()
    #     val_loss/=iter




