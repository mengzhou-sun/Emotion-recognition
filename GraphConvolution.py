from dgl.nn.pytorch.conv import GraphConv
import numpy as np
import torch.nn as nn
import torch
import dgl
#graph definition
# u,v=torch.tensor([[4,3,7,6,13,12,10,9,11,8,5,2,0,15,14,17,16],[3,2,6,5,12,11,9,8,5,2,1,1,1,0,0,15,14]])
# g=dgl.graph((u,v))
# bg = dgl.to_bidirected(g)
# c=c.permute(1,0)
# # #add feature(x,y)
# bg.ndata['feat']=c
class GCN1(nn.Module):
    def __init__(self, in_feats, h_feats, hidden_dim):
        super(GCN1, self).__init__()
        super().__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, hidden_dim)


    def forward(self, g,in_feat):

        h = self.conv1(g, in_feat)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


#predict=model_Gcn(bg,bg.ndata['feat'].float())
#print(predict)
def relativeMiddleCor(x_list, y_list):
        # 计算相对于几何中心的坐标

        # 计算几何中心坐标
        min_x = min(x_list)
        max_x = max(x_list)

        min_y = min(y_list)
        max_y = max(y_list)



        middle_p_x = min_x+ 0.5*(max_x-min_x)
        middle_p_y = min_y+ 0.5*(max_y-min_y)


        # p(相对) = (x原始 -  Px(重心), y原始 -  Py(重心))
        x_list = np.array(x_list) - middle_p_x
        y_list = np.array(y_list) - middle_p_y


        x_y_column = np.column_stack((x_list, y_list))

        return x_y_column

