import os
import torch.utils.data as data
import torch
import numpy as np
import json

path = "/home/kun/Downloads/Pointnet_Pointnet2_pytorch-master/data/ShapeNet"


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 存标签和文件夹关系
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:  # 只读
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]     # 将标签与文件夹信息写进字典
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}   # 将字典关系调换

        self.meta = {}
        # .format() 中是用来填入{}的字符
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed() 加载打乱的
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        # 这里file指的是打乱后的文件地址，种类即'Airplane': '02691156'中的数字，uuid是文件名
        for file in filelist:
            _, category, uuid = file.split('/')
            # 这里的判断主要是因为有些情况下剔除了部分种类，
            # 然后建立了两个新的文件目录，一个是 points 文件夹及 .pts文件，一个是 points_label 文件夹及 .seg文件
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        # cat里存的是标签与文件夹名字的对应关系
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
        # 给标签赋予对应的0-15数字表示
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), './misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample 2500个
        point_set = point_set[choice, :]
        # 归一化：减去x,y,z轴上平均值，再除以范围。而这个范围的度量使用的是每个点离原点的距离
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)
