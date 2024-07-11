import numpy as np
import scipy.io as sio
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from sklearn import preprocessing
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

def map_att(seen_att, label, classes, opt):
    mapped_att = torch.FloatTensor(label.shape[0], opt.attSize)
    for i in range(classes.size(0)):
        mapped_att[label == classes[i]] = seen_att[i]
    return mapped_att

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))
    MI_loss = (torch.mean(kl_divergence) - i_c)
    return MI_loss

def optimize_beta(beta, MI_loss, alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))
    return beta_new

def gradient_penalty_d(netD, real_data, fake_data, device, opt):
    # print real_data.size()
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(device)

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size()).to(device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda0
    return gradient_penalty

def gradient_penalty_d_RF(netD, real_data, fake_data, device, opt):
    # print real_data.size()
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(device)

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates, _, _ = netD(interpolates)

    ones = torch.ones(disc_interpolates.size()).to(device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda0
    return gradient_penalty

def generate_syn_feature(netG, netE, classes, attribute, device, opt, syn_num):

    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * syn_num, opt.resSize)
    syn_label = torch.LongTensor(nclass * syn_num)
    syn_att = torch.FloatTensor(syn_num, opt.attSize)
    syn_noise = torch.FloatTensor(syn_num, opt.noiseSize)

    syn_att = syn_att.to(device)
    syn_noise = syn_noise.to(device)

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(syn_num, 1))
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        emb_output = netE(output, emb=True)
        syn_feature.narrow(0, i * syn_num, syn_num).copy_(emb_output.data.cpu())
        syn_label.narrow(0, i * syn_num, syn_num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_aug_feature_after(netG, netE, classes, attribute, device, opt, syn_num, dataset):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * syn_num, opt.hidden_size)
    syn_feature_no_mix = torch.FloatTensor(nclass * syn_num, opt.hidden_size)
    syn_proj_feature_no_mix = torch.FloatTensor(nclass * syn_num, opt.outzSize)
    mix_fea = torch.FloatTensor(syn_num, opt.hidden_size)
    syn_label = torch.LongTensor(nclass * syn_num)
    syn_att = torch.FloatTensor(syn_num, opt.attSize)
    syn_noise = torch.FloatTensor(syn_num, opt.noiseSize)

    syn_att = syn_att.to(device)
    syn_noise = syn_noise.to(device)

    embed_feature = netE(dataset.train_feature.cuda(), emb=True)  # real embed
    tr_cls_centroid = torch.zeros([dataset.seenclass_num, dataset.train_feature.shape[1]])
    for j in range(dataset.seenclass_num):
        tr_cls_centroid[j] = torch.mean(embed_feature[dataset.train_local_label == j], dim=0)
    for i in range(nclass):
        weight = F.softmax(dataset.s_u_semantic_similarity_unseen[i], dim=-1)
        closer_seen_center = tr_cls_centroid[dataset.s_unseen_idx_mat[i]]  # 与未见类相似度最高的N个可见类特征中心
        calib_fea = torch.einsum('i,ik->k', [weight, closer_seen_center.data.cpu()])
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(syn_num, 1))
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        emb_output, outz = netE(output, retrieval=True)  # syn embed
        for j in range(syn_num):
           mix_fea[j] = opt.gamma * emb_output[j] + (1 - opt.gamma) * calib_fea.cuda()

        syn_feature.narrow(0, i * syn_num, syn_num).copy_(mix_fea.data.cpu())
        syn_feature_no_mix.narrow(0, i * syn_num, syn_num).copy_(emb_output.data.cpu())
        syn_proj_feature_no_mix.narrow(0, i * syn_num, syn_num).copy_(outz.data.cpu())
        syn_label.narrow(0, i * syn_num, syn_num).fill_(iclass)

    return syn_feature, syn_feature_no_mix, syn_proj_feature_no_mix, syn_label

def generate_syn_aug_feature_after_RF(netG, netE, classes, attribute, device, opt, syn_num, dataset):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * syn_num, opt.hidden_size)
    mix_fea = torch.FloatTensor(syn_num, opt.hidden_size)
    syn_label = torch.LongTensor(nclass * syn_num)
    syn_att = torch.FloatTensor(syn_num, opt.attSize)
    syn_att_all = torch.FloatTensor(nclass * syn_num, opt.attSize)
    syn_noise = torch.FloatTensor(syn_num, opt.noiseSize)

    syn_att = syn_att.to(device)
    syn_noise = syn_noise.to(device)

    embed_feature = netE(dataset.train_feature.cuda(), emb=True)  # real embed
    tr_cls_centroid = torch.zeros([dataset.seenclass_num, dataset.train_feature.shape[1]])
    for j in range(dataset.seenclass_num):
        tr_cls_centroid[j] = torch.mean(embed_feature[dataset.train_local_label == j], dim=0)
    for i in range(nclass):
        weight = F.softmax(dataset.s_u_semantic_similarity_unseen[i], dim=-1)
        closer_seen_center = tr_cls_centroid[dataset.s_unseen_idx_mat[i]]  # 与未见类相似度最高的N个可见类特征中心
        calib_fea = torch.einsum('i,ik->k', [weight, closer_seen_center.data.cpu()])
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(syn_num, 1))
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        emb_output = netE(output, emb=True)  # syn embed
        for j in range(syn_num):
           mix_fea[j] = opt.gamma * emb_output[j] + (1 - opt.gamma) * calib_fea.cuda()

        syn_feature.narrow(0, i * syn_num, syn_num).copy_(mix_fea.data.cpu())
        syn_label.narrow(0, i * syn_num, syn_num).fill_(iclass)
        syn_att_all.narrow(0, i * syn_num, syn_num).copy_(syn_att)

    return syn_feature, syn_label, syn_att_all
