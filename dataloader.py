import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

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
        mat = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        self.feature = mat['features'].T
        self.label = mat['labels'].astype(int).squeeze() - 1
        self.image_file = mat['image_files']

        mat = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")

        trainval_loc = mat['trainval_loc'].squeeze() - 1
        test_seen_loc = mat['test_seen_loc'].squeeze() - 1
        test_unseen_loc = mat['test_unseen_loc'].squeeze() - 1
        self.allclasses = torch.from_numpy(np.unique(self.label))
        self.allclasses_num = self.allclasses.size(0)

        if opt.dataset == 'CUB':
            file_path = opt.dataroot + '/CUB/sent_splits.mat' # 1024D
            mat = sio.loadmat(file_path)
            self.attribute = F.normalize(torch.from_numpy(mat['att'].T), dim=1).float()
            # file_path = opt.dataroot + '/CUB/cub_attributes_reed.npy'
            # self.attribute = F.normalize(torch.from_numpy(np.load(file_path)), dim=1).float()
        else:
            self.attribute = torch.from_numpy(mat['att'].T).float()

        if opt.fine_tuning:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                _train_feature = scaler.fit_transform(np.load(f'fine_tuning_data/{opt.dataset}/train_feature.npy'))
                _test_seen_feature = scaler.fit_transform(np.load(f'fine_tuning_data/{opt.dataset}/test_seen_feature.npy'))
                _test_unseen_feature = scaler.fit_transform(np.load(f'fine_tuning_data/{opt.dataset}/test_unseen_feature.npy'))

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)                                                     # train seen feature
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)                                               # test unseen feature
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)                                                 # test seen feature
            else:
                self.train_feature = torch.from_numpy(np.load(f'fine_tuning_data/{opt.dataset}/train_feature.npy'))
                self.test_seen_feature = torch.from_numpy(np.load(f'fine_tuning_data/{opt.dataset}/test_seen_feature.npy'))
                self.test_unseen_feature = torch.from_numpy(np.load(f'fine_tuning_data/{opt.dataset}/test_unseen_feature.npy'))

            self.train_label = torch.from_numpy(np.load(f'fine_tuning_data/{opt.dataset}/train_label.npy'))
            self.test_seen_label = torch.from_numpy(np.load(f'fine_tuning_data/{opt.dataset}/test_seen_label.npy'))
            self.test_unseen_label = torch.from_numpy(np.load(f'fine_tuning_data/{opt.dataset}/test_unseen_label.npy'))
        else:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()     # sklearn库中的数据预处理函数（基于numpy）
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(self.feature[trainval_loc])
                _test_seen_feature = scaler.transform(self.feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(self.feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)                                                     # train seen feature
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)                                               # test unseen feature
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)                                                 # test seen feature
            else:
                self.train_feature = torch.from_numpy(self.feature[trainval_loc]).float()           # train seen feature
                self.test_unseen_feature = torch.from_numpy(self.feature[test_unseen_loc]).float()  # test unseen feature
                self.test_seen_feature = torch.from_numpy(self.feature[test_seen_loc]).float()      # test seen feature

            self.train_label = torch.from_numpy(self.label[trainval_loc]).long()             # train seen label
            self.test_seen_label = torch.from_numpy(self.label[test_seen_loc]).long()        # test seen label
            self.test_unseen_label = torch.from_numpy(self.label[test_unseen_loc]).long()    # test unseen label
            self.test_unseen_image_file = self.image_file[test_unseen_loc]

        self.ntrain = self.train_feature.shape[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))            # seen classes
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))    # unseen classes
        self.ntest_class = self.unseenclasses.shape[0]
        self.train_local_label = map_label(self.train_label, self.seenclasses)                 # train local label
        self.test_seen_local_label = map_label(self.test_seen_label, self.seenclasses)         # test seen local label
        self.test_unseen_local_label = map_label(self.test_unseen_label, self.unseenclasses)   # test unseen local label

        self.seenclass_num = self.seenclasses.size(0)       # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)   # number of unseen classes

        self.seen_att = self.attribute[self.seenclasses]        # attribute of seen classes
        self.unseen_att = self.attribute[self.unseenclasses]    # attribute of unseen classes
        self.reset_att = torch.cat((self.seen_att, self.unseen_att), dim=0)
        self.feature_dim = self.train_feature.shape[1]                  # dim of feature
        self.att_dim = self.attribute.shape[1]                          # dim of attribute
        self.train_data = [self.train_feature.numpy(), self.train_label.numpy()]
        self.test_data = [self.test_unseen_feature.numpy(), self.test_unseen_label.numpy()]
        self.class_feature = []
        self.class_label = []
        self.s_u_semantic_similarity_check(self.opt.seen_Neighbours, self.seen_att, self.unseen_att)

        # collect the data of each class
        self.tr_cls_centroid = torch.zeros([self.seenclass_num, self.train_feature.shape[1]])
        for i in range(self.seenclass_num):
            self.tr_cls_centroid[i] = torch.mean(self.train_feature[self.train_local_label == i], dim=0)
        self.seen_class_num = []
        for i in self.seenclasses:
            num = self.train_feature[self.train_label == i].shape[0]
            self.seen_class_num.append(num)
        self.seen_class_num = np.array(self.seen_class_num)

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

