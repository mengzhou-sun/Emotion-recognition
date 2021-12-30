import numpy as np

def compute_pck(dt_kpts, gt_kpts, refer_kpts):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,k]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,k]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　
    :return: 相关指标
    """

    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)
    assert (len(refer_kpts) == 2)
    assert (dt.shape[0] == gt.shape[0])
    ranges = np.arange(0.0, 0.1, 0.01)
    kpts_num = gt.shape[2]
    ped_num = gt.shape[0]

    # compute dist
    scale = np.sqrt(np.sum(np.square(gt[:, :, refer_kpts[0]] - gt[:, :, refer_kpts[1]]), 1))
    dist = np.sqrt(np.sum(np.square(dt - gt), 1)) / np.tile(scale, (gt.shape[2], 1)).T

    # compute pck
    pck = np.zeros([ranges.shape[0], gt.shape[2] + 1])
    for idh, trh in enumerate(list(ranges)):
        for kpt_idx in range(kpts_num):
            pck[idh, kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= trh)
        # compute average pck
        pck[idh, -1] = 100 * np.mean(dist <= trh)
    return pck




keypoint_Alpha_prediction=[(514, 73), (514, 156), (486, 135), (403, 198), (445, 135), (542, 177), (473, 281), (431, 208), (542, 333), (403, 354), None, (570, 344), (445, 406), None, (542, 271)]
key_point_coco_Groundtruth= [469, 142, 2, 485, 137, 2, 469, 125, 2, 519, 147, 2, 0, 0, 0, 565, 213, 2, 518, 163, 2, 478, 286, 2, 412, 203, 2, 437, 201, 2, 455, 144, 2, 557, 345, 2, 511, 324, 1, 446, 419, 2, 394, 354, 2, 0, 0, 0, 460, 422, 2]
keypoint_coco_prediction=   [459, 135 ,528, 177, 500, 166,  445, 146, 556, 187, 473, 281, 431, 198, 514, 323, 389, 354, 431, 417, 556, 344, 431, 406, 0,0, 459, 125, 473, 135, 0,0, 514, 135]
print(len(key_point_coco_Groundtruth))
sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
variances = (sigmas * 2) ** 2


def compute_kpts_oks(dt_kpts, gt_kpts, area):


    g = np.array(gt_kpts)
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    assert (np.count_nonzero(vg > 0) > 0)
    d = np.array(dt_kpts)
    xd = d[0::2]
    yd = d[1::2]

    dx = xd - xg
    dy = yd - yg

    e = (dx ** 2 + dy ** 2) / variances / (area + np.spacing(1)) / 2
    print(e)
    e = e[vg > 0]

    return (np.sum(e)) / e.shape[0]

print(compute_kpts_oks(keypoint_coco_prediction,key_point_coco_Groundtruth,1))

