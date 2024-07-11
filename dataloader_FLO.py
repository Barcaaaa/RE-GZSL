import numpy as np
import random
import scipy.io as sio
import torch
import time
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from sklearn import preprocessing
#from git import clip
from sklearn.metrics.pairwise import cosine_similarity
import torch.utils.data as data
import torch.nn.functional as F
from metric.utils import pdist
from tqdm import tqdm
import numpy

# glob label to local label
def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

# load dataset
class Dataset(object):
    def __init__(self, opt):
        self.opt = opt
        if opt.dataset in ['FLO']:
            self.read(opt)
        else:
            self.read_matdataset(opt)
    def read(self, opt):
        if opt.dataset == "FLO":
            opt.dataset = "FLO"

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        train_att = matcontent['att_train']
        seen_pro = matcontent['seen_pro']
        attribute = matcontent['attribute']
        unseen_pro = matcontent['unseen_pro']
        self.attribute = torch.from_numpy(attribute).float()
        self.train_att = torch.from_numpy(seen_pro.astype(np.float32)).float()
        self.test_att = torch.from_numpy(unseen_pro.astype(np.float32))

        train_fea = matcontent['train_fea']
        test_seen_fea = matcontent['test_seen_fea']
        test_unseen_fea = matcontent['test_unseen_fea']

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(train_fea)
        _test_seen_feature = scaler.transform(test_seen_fea)
        _test_unseen_feature = scaler.transform(test_unseen_fea)
        mx = _train_feature.max()
        train_fea = train_fea * (1 / mx)
        test_seen_fea = test_seen_fea * (1 / mx)
        test_unseen_fea = test_unseen_fea * (1 / mx)

        self.train_feature = torch.from_numpy(train_fea).float()
        self.test_seen_feature = torch.from_numpy(test_seen_fea).float()
        self.test_unseen_feature = torch.from_numpy(test_unseen_fea).float()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/label.mat")

        train_idx = matcontent['train_idx'] - 1
        train_label = matcontent['train_label_new']
        test_unseen_idex = matcontent['test_unseen_idex'] - 1
        test_seen_idex = matcontent['test_seen_idex'] - 1
        self.train_label = torch.from_numpy(train_idx.squeeze()).long()
        self.test_seen_label = torch.from_numpy(test_seen_idex.squeeze()).long()
        self.test_unseen_label = torch.from_numpy(test_unseen_idex.squeeze()).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.seenclass_num = self.seenclasses.size(0)  # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)
        self.train_local_label = self.train_label
        self.s_u_semantic_similarity_check(self.opt.seen_Neighbours, self.train_att, self.test_att)
        self.seen_att = self.train_att
        self.unseen_att = self.test_att
        self.test_seen_local_label = map_label(self.test_seen_label, self.seenclasses)  # test seen local label
        self.test_unseen_local_label = map_label(self.test_unseen_label, self.unseenclasses)  # test unseen local label

        self.seenclass_num = self.seenclasses.size(0)  # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)  # number of unseen classes
        self.allclasses_num = self.allclasses.size(0)       # number of all classes
        self.feature_dim = self.train_feature.shape[1]  # dim of feature
        self.att_dim = self.attribute.shape[1]  # dim of attribute
        self.train_data = [self.train_feature.numpy(), self.train_label.numpy()]
        # self.aug_data = [aug_features.numpy(),self.train_local_label.numpy()]
        self.test_data = [self.test_unseen_feature.numpy(), self.test_unseen_label.numpy()]
        self.class_feature = []
        self.class_label = []
        self.tr_cls_centroid = torch.zeros([self.seenclass_num, self.train_feature.shape[1]])
        for i in range(self.seenclass_num):
            self.tr_cls_centroid[i] = torch.mean(self.train_feature[self.train_local_label == i], dim=0)

    def s_u_semantic_similarity_check(self, Neighbours, train_text_feature, test_text_feature):
        '''
        Unseen class
        '''
        unseen_similarity_matric = torch.from_numpy(cosine_similarity(test_text_feature, train_text_feature))

        # Mapping matric
        self.s_unseen_idx_mat = torch.argsort(-1 * unseen_similarity_matric, dim=1)
        self.s_unseen_idx_mat = self.s_unseen_idx_mat[:, 0:Neighbours]

        # Neighbours Semantic similary values
        self.s_u_semantic_similarity_unseen = torch.zeros((self.unseenclass_num, Neighbours))
        for i in range(self.unseenclass_num):
            for j in range(Neighbours):
                self.s_u_semantic_similarity_unseen[i, j] = unseen_similarity_matric[i, self.s_unseen_idx_mat[i, j]]













