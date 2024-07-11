import os
import sys
import time
import datetime
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from option import Options
import zsl
import utils
import model
from metric.loss import RkdDistance, RKdAngle

from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)

    def forward(self, x):
        o = self.fc(x)
        return o

def val(test_X, netE, netC, opt):
    start = 0
    ntest = test_X.size()[0]
    predicted_label = torch.LongTensor(test_X.size(0))
    for i in range(0, ntest, opt.c_batch_size):
        end = min(ntest, start + opt.c_batch_size)
        embed = netE(test_X[start:end].to(device), emb=True)
        output = netC(embed).cpu()
        predicted_label[start:end] = torch.max(output, 1)[1]
        start = end
    return predicted_label

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

opt = Options().parse()
# os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus)

if opt.dataset == 'FLO':
    import classifier_embed_FLO as classifier
    import dataloader_FLO as dataloader
else:
    import classifier_embed as classifier
    import dataloader

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
month = datetime.datetime.now().month
day = datetime.datetime.now().day
hour = datetime.datetime.now().hour
date = str(month)+'_'+str(day)+'_'+str(hour)
start_time = GetNowTime()
print(start_time)
print('Begin run!!!')

print(opt)
sys.stdout.flush()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)  # if you are using multi-GPU.
np.random.seed(opt.manualSeed)  # Numpy module.
torch.manual_seed(opt.manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = False
cudnn.deterministic = True

if opt.cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
cudnn.benchmark = True

if not torch.cuda.is_available() and not opt.cuda:
    print("WARNING: No GPU!")
    exit()

# load data
dataset = dataloader.Dataset(opt)

opt.attSize = dataset.att_dim
print('attSize:', opt.attSize)
opt.noiseSize = dataset.att_dim
print('noiseSize:', opt.noiseSize)
opt.resSize = dataset.feature_dim
opt.nclass = dataset.allclasses_num
opt.class_num = dataset.allclasses_num
opt.unseen_num = dataset.unseenclass_num
opt.seen_num = dataset.seenclass_num

print("Training samples: ", dataset.train_label.shape[0])

baseline = True

# eval
netG = model.Generator(opt).to(device)
netC = SoftmaxClassifier(opt.hidden_size, opt.class_num).to(device)

if baseline:
    model_path = './output/baseline'
    netE = model.Embedding_model_baseline(opt, dataset).to(device)
else:
    model_path = './output/RE-GZSL'
    netE = model.Embedding_model(opt, dataset).to(device)

state_dict = torch.load(os.path.join(model_path, 'best_model.pth'))
loaded_parameters = netG.load_state_dict(state_dict.pop('state_dict_G'))
# if not loaded_parameters.missing_keys and not loaded_parameters.unexpected_keys:
#     print("G success！")
loaded_parameters = netE.load_state_dict(state_dict.pop('state_dict_E'))
# if not loaded_parameters.missing_keys and not loaded_parameters.unexpected_keys:
#     print("E success！")
loaded_parameters = netC.load_state_dict(state_dict.pop('state_dict_C'))
# if not loaded_parameters.missing_keys and not loaded_parameters.unexpected_keys:
#     print("C success！")
# print(netE.state_dict().keys())

netG.eval()
netE.eval()

for p in netE.parameters():  # reset requires_grad
    p.requires_grad = False

test_seen_label = dataset.test_seen_local_label
test_unseen_label = dataset.test_unseen_local_label + dataset.seenclass_num

pred_unseen = val(dataset.test_unseen_feature, netE, netC, opt)
acc_unseen = torch.sum(pred_unseen == test_unseen_label).float() / \
             float(test_unseen_label.size(0))
pred_seen = val(dataset.test_seen_feature, netE, netC, opt)
acc_seen = torch.sum(pred_seen == test_seen_label).float() / \
           float(test_seen_label.size(0))
H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

if baseline:
    if opt.dataset == 'FLO':
        syn_feature, syn_label = utils.generate_syn_feature(
            netG, netE, dataset.unseenclasses, dataset.test_att, device, opt, opt.syn_num)
    else:
        syn_feature, syn_label = utils.generate_syn_feature(
            netG, netE, dataset.unseenclasses, dataset.attribute, device, opt, opt.syn_num)
else:
    if opt.dataset == 'FLO':
        syn_feature, syn_feature_no_mix, syn_proj_feature_no_mix, syn_label = utils.generate_syn_aug_feature_after(
            netG, netE, dataset.unseenclasses, dataset.test_att, device, opt, opt.syn_num, dataset)
    else:
        syn_feature, syn_feature_no_mix, syn_proj_feature_no_mix, syn_label = utils.generate_syn_aug_feature_after(
            netG, netE, dataset.unseenclasses, dataset.attribute, device, opt, opt.syn_num, dataset)

# 训练时是保存最好的分类模型，而非检索最好的
retrieval_list = []
embed_feature = netE(dataset.test_unseen_feature.cuda(), emb=True)
cls_centroid = np.zeros((dataset.ntest_class, opt.resSize))
for i, class_idx in enumerate(dataset.unseenclasses):
    if baseline:
        cls_centroid[i] = torch.mean(syn_feature[syn_label == class_idx], dim=0)
    else:
        cls_centroid[i] = torch.mean(syn_feature_no_mix[syn_label == class_idx], dim=0)

dist = cosine_similarity(cls_centroid, embed_feature.cpu())

precision_100 = torch.zeros(dataset.ntest_class)
precision_50 = torch.zeros(dataset.ntest_class)
precision_25 = torch.zeros(dataset.ntest_class)

dist = torch.from_numpy(-dist)
for i, class_idx in enumerate(dataset.unseenclasses):
    is_class = dataset.test_unseen_label == class_idx
    cls_num = int(is_class.sum())

    # 100%
    value, idx = torch.topk(dist[i, :], cls_num, largest=False)
    precision_100[i] = (is_class[idx]).sum().float() / cls_num

    # 50%
    cls_num_50 = int(cls_num / 2)
    _, idx = torch.topk(dist[i, :], cls_num_50, largest=False)
    precision_50[i] = (is_class[idx]).sum().float() / cls_num_50

    # 25%
    cls_num_25 = int(cls_num / 4)
    _, idx = torch.topk(dist[i, :], cls_num_25, largest=False)
    precision_25[i] = (is_class[idx]).sum().float() / cls_num_25

print(f'dataset: {opt.dataset}, seed: {opt.manualSeed}')
print(f'unseen:{round(acc_unseen.item(), 3)}, seen:{round(acc_seen.item(), 3)}, h:{round(H.item(), 3)}')
print("retrieval results: 100%%: %.3f, 50%%: %.3f, 25%%: %.3f" % (
    precision_100.mean().item(), precision_50.mean().item(), precision_25.mean().item()))

tsne_flag = True
if tsne_flag:
    ### unseen
    sample_size = 2000

    index_1 = np.random.choice(embed_feature.shape[0], size=sample_size, replace=False)
    index_2 = np.random.choice(syn_feature.shape[0], size=sample_size, replace=False)
    
    total_feature = np.concatenate((embed_feature[index_1].cpu().numpy(), syn_feature[index_2].cpu().numpy()))

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(total_feature)

    plt.figure(figsize=(10, 10))

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'gray']
    # c=colors[dataset.test_unseen_local_label[i]]
    
    center_points = []
    for label in np.unique(dataset.test_unseen_local_label):
        center_point = np.mean(X_tsne[:sample_size][dataset.test_unseen_local_label[index_1] == label], axis=0)
        center_points.append(center_point)
    for i, center_point in enumerate(center_points):
        label = str(i)
        plt.annotate(label, (center_point[0], center_point[1]), ha='center', fontsize=16)

    center_points = []
    syn_test_unseen_local_label = map_label(syn_label, dataset.unseenclasses)
    for label in np.unique(syn_test_unseen_local_label):
        center_point = np.mean(X_tsne[sample_size:][syn_test_unseen_local_label[index_2] == label], axis=0)
        center_points.append(center_point)
    for i, center_point in enumerate(center_points):
        label = str(i)
        plt.annotate(label, (center_point[0], center_point[1]), ha='center', fontsize=18)
    
    for i in range(len(X_tsne)):
        if i < sample_size:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='steelblue', alpha=0.5)
        else:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='lightseagreen', alpha=0.5)

    names = ['real sample', 'synthetic sample']
    colors = ['steelblue', 'lightseagreen']
    # 太多点会报错，构建一个空图例
    handles = [plt.scatter([], [], color=colors[i], label=names[i]) for i in range(len(names))]
    plt.legend(handles, names, fontsize=20, loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(os.path.join(model_path, 'tsne_refine.png'), dpi=300, bbox_inches='tight')
    plt.clf()

    ### seen
    sample_size = 2000
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'gold', 'gray', 'orange', 'purple', 'brown'] * 4
    alpha = [0.2]*10
    alpha.extend([0.4]*10)
    alpha.extend([0.6]*10)
    alpha.extend([0.8]*10)

    embed_feature_seen = netE(dataset.test_seen_feature.cuda(), emb=True).cpu().numpy()
    print(embed_feature_seen.shape)
    index = np.random.choice(embed_feature_seen.shape[0], size=sample_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embed_feature_seen[index])

    plt.figure(figsize=(10, 10))

    for i in range(len(X_tsne)):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=colors[dataset.test_seen_local_label[index][i]], 
            alpha=alpha[dataset.test_seen_local_label[index][i]])

    plt.grid(True, alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(os.path.join(model_path, 'tsne_seen.png'), dpi=300, bbox_inches='tight')
    plt.clf()

sys.stdout.flush()
