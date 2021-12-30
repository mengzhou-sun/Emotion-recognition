import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
from torch.utils.data import Dataset, DataLoader
import json
from numpy import random
import torchvision.transforms.functional as transforms
from pathlib import Path
import skimage.io as io
from utils import *
from PIL import Image

from posenet import *
import dgl

class EmoticDataset(Dataset):
    def __init__(self,  mode, transforms1=False):

        super(EmoticDataset, self).__init__()

        self.transforms1 = transforms1
        with open(Path( "emotion_" + mode + "_EMOTICAG.json")) as f:
            emotion = json.load(f)
            emotions = organize_array(emotion, "emotions")
            bbox = organize_array(emotion, "bbox")
            cont_variable = organize_array(emotion, "continuous_variables")
            img_url = organize_array(emotion, "coco_url", origin="images")
            image_name = organize_array(emotion, "file_name", origin="images")
            self.lbl = emotions
            self.cont_variable = cont_variable
            self.bbox = bbox


            self.image_url = img_url
            self.image_name = image_name

    def __len__(self):

            return int(0.5*len(self.lbl))

    def __getitem__(self, index):
            global image_context
            try:
                image = io.imread(Path(".", "Dataset", "Images", self.image_name[index][0]))
            except:
                image = io.imread(self.image_url[index][0])

            #image = io.imread(self.image_url[index][0])
            #image = image / np.max(image)
            if len(image.shape) < 3:
                image = np.stack((image,) * 3, axis=-1)

            bbox=self.bbox[index]
            label=self.lbl[index]
            cv2.rectangle((image), (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[3]) + int(bbox[1])),
                          (0, 0, 0), 1)
            image1=image.copy()
            cv2.rectangle((image1), (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0]) + int(bbox[2]), int(bbox[3]) + int(bbox[1])),
                          (0, 0, 0), -1)
            image1=Image.fromarray(image1)
            image=Image.fromarray((image))
            image_body=image.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[3]+bbox[1]))


            image_body=image_body.resize((64,128))
            image3=image1.resize((128,128))
            image_body3= cv2.cvtColor(np.asarray(image_body), cv2.COLOR_BGR2RGB)
            pose_array=openpose_net(image_body3)
            image4=np.array(image3)
            image_body4=np.array(image_body3)
            image_context = image4.astype(np.float32)
            image_body5 = image_body4.astype(np.float32)
            pose_array = pose_array.astype(np.int64)
            pose_array = torch.from_numpy(pose_array)

            # pose_array = pose_array.permute(1, 2, 0)
            # pose_array = torch.squeeze(pose_array, dim=2)
            #
            # u, v = torch.tensor([[4, 3, 7, 6, 13, 12, 10, 9, 11, 8, 5, 2, 0, 15, 14, 17, 16],
            #                      [3, 2, 6, 5, 12, 11, 9, 8, 5, 2, 1, 1, 1, 0, 0, 15, 14]])
            # g = dgl.graph((u, v))
            # bg = dgl.to_bidirected(g)
            # bg.ndata['feat'] = pose_array


            image_context, image_body= torch.from_numpy(image_context), torch.from_numpy(
                image_body5)

            # if self.transforms1 ==False:
            #
            #    return image_context,image_body,pose_array,label
            # if self.transforms1 ==True:
            #
            #     if random.random() > 0.5:
            #         image_context=transforms.gaussian_blur(image_context,[3,3])
            #         image_body=transforms.gaussian_blur(image_body,[3,3])

            return image_context,image_body,pose_array,label




dataset = EmoticDataset(mode="train", transforms1=True)
validation_loader = DataLoader(dataset=dataset,batch_size=1,
                                    shuffle=True,)





from dgl.nn.pytorch.conv import GraphConv

# #graph definition
# u,v=torch.tensor([[4,3,7,6,13,12,10,9,11,8,5,2,0,15,14,17,16],[3,2,6,5,12,11,9,8,5,2,1,1,1,0,0,15,14]])
# g=dgl.graph((u,v))
# bg = dgl.to_bidirected(g)
# c=c.permute(1,0)
# #add feature(x,y)
# bg.ndata['feat']=c
# print(bg)
# class GCN1(nn.Module):
#     def __init__(self, in_feats, h_feats, hidden_dim):
#         super(GCN1, self).__init__()
#         super().__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, hidden_dim)
#
#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = torch.relu(h)
#         h = self.conv2(g, h)
#         h = torch.relu(h)
#
#
#         return h
#
# model_Gcn=GCN1(2,18,9)
# predict=model_Gcn(bg,bg.ndata['feat'].float())
# print(predict)