import utils
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from metric.loss import RkdDistance, RKdAngle
import sys
import model
import numpy as np
import time
from option import Options
import datetime
import zsl
import os
import math
from sklearn.metrics.pairwise import cosine_similarity


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
torch.cuda.manual_seed_all(opt.manualSeed)  # if you are using multi-GPU
np.random.seed(opt.manualSeed)
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
criterion = nn.CrossEntropyLoss().to(device)
BCE = torch.nn.BCEWithLogitsLoss().to(device)
input_res = torch.FloatTensor(opt.way * 2 * opt.shot, opt.resSize).to(device)
input_att = torch.FloatTensor(opt.way * 2 * opt.shot, opt.attSize).to(device)
noise = torch.FloatTensor(opt.way * 2 * opt.shot, opt.noiseSize).to(device)
input_label = torch.LongTensor(opt.way * 2 * opt.shot).to(device)
local_label  = torch.LongTensor(opt.way * 2 * opt.shot).to(device)
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)

def sample(x_spt,y_spt):
    x_spt = torch.from_numpy(x_spt)
    y_spt = torch.from_numpy(y_spt)
    input_res.copy_(x_spt)
    input_att.copy_(dataset.attribute[y_spt])
    input_label.copy_(y_spt)
    if opt.dataset == 'FLO':
        input_att.copy_(dataset.train_att[y_spt])
        local_label.copy_(y_spt)
    else:
        input_att.copy_(dataset.attribute[y_spt])
        local = utils.map_label(y_spt, dataset.seenclasses)
        local_label.copy_(local)

db = zsl.zsl_NShot(dataset.train_data, dataset.test_data, batchsz=opt.critic_iter, 
                   n_way=opt.way, k_shot=opt.shot, args=opt)

print("Training samples: ", dataset.train_label.shape[0])

# initialize generator and discriminator
print(dataset.attribute.shape[1])
netG = model.Generator(opt).to(device)
netD = model.Discriminator(opt).to(device)
netE = model.Embedding_model_baseline(opt, dataset).to(device)
# setup optimize
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=3e-4)
dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
best_seen_acc = 0.0
best_unseen_acc = 0.0
best_h = 0.0
best_epoch = 0
best_recall_1_s = 0
best_recall_1_u = 0
c_epoch = 0
best_100_mAP = 0.0
best_50_mAP = 0.0
best_25_mAP = 0.0
best_retrieval_epoch = 0
best_retrieval_list = []

for epoch in range(opt.epoch):
    since = time.time()
    print('EP[%d/%d]******************************************************' % (epoch, opt.epoch))
    #netE.get_center()
    for i in range(0, dataset.train_label.shape[0], opt.way * 2 * opt.shot):
        since_e = time.time()
        x_spt, y_spt = db.next('train')

        # Step1: train discriminator
        for p in netD.parameters():
            p.requires_grad = True
        for p in netE.parameters():  # reset requires_grad
            p.requires_grad = True
        for iter_d in range(opt.critic_iter):
            sample(x_spt[iter_d], y_spt[iter_d])
            netD.zero_grad()

            real_data = torch.cat((input_res, input_att), 1)
            criticD_real = netD(real_data)
            criticD_real = -opt.gammaD * criticD_real.mean()
            criticD_real.backward()

            noise.normal_(0, 1)
            fake = netG(noise, input_att)
            fake_data = torch.cat((fake.detach(), input_att), dim=1)
            criticD_fake = netD(fake_data)
            criticD_fake = opt.gammaD * criticD_fake.mean()
            criticD_fake.backward()

            gp_loss = utils.gradient_penalty_d(netD, real_data, fake_data, device, opt)
            gp_loss = opt.gammaD * gp_loss
            gp_loss.backward()

            optimizerD.step()
            since_dp = time.time()
            loss_cls = netE(input_res, label=input_label, local_label=local_label)

            loss =  opt.embed_ratio * loss_cls

            optimizerE.zero_grad()
            loss.backward()
            # time_dp = time.time() - since_dp
            # print('End dp!!!')
            # print('Time Elapsed: {}'.format(time_dp))
            optimizerE.step() # update moving average of target encoder

        # Step2: train generator
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netE.parameters():  # reset requires_grad
            p.requires_grad = False

        netG.zero_grad()
        noise.normal_(0, 1)
        fake = netG(noise, input_att)

        criticG_fake = netD(torch.cat((fake, input_att), dim=1))

        loss_fake = netE(fake, label=input_label, local_label = local_label)

        loss_G_E = opt.embed_ratio * loss_fake

        D_loss = -criticG_fake.mean()

        G_loss = opt.gammaG * D_loss + opt.E_weight * loss_G_E
        G_loss.backward()

        optimizerG.step()
        # time_elapsed = time.time() - since
        # print('End iter!!!')
        # print('Time Elapsed: {}'.format(time_elapsed))
    netG.eval()

    # Step3: train classifier
    # initialize
    if epoch >= opt.best_epoch:
        for p in netE.parameters():  # reset requires_grad
            p.requires_grad = False
        if opt.dataset == 'FLO':
            syn_feature, syn_label = utils.generate_syn_feature(
                netG, netE, dataset.unseenclasses, dataset.test_att, device, opt, opt.syn_num)
        else:
            syn_feature, syn_label = utils.generate_syn_feature(
                netG, netE, dataset.unseenclasses, dataset.attribute, device, opt, opt.syn_num)

        Cls = classifier.Classifier(syn_feature, syn_label, dataset, netE, opt, epoch, generalized=True)

        if Cls.h >= best_h:
            best_h = Cls.h
            best_unseen_acc = Cls.unseen_acc
            best_seen_acc = Cls.seen_acc
            # c_epoch = Cls.epoch
            best_epoch = epoch
            out_dir = './output/baseline/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save({'epoch': epoch, 
                        'state_dict_E': netE.state_dict(),
                        'state_dict_G': netG.state_dict(),
                        'state_dict_C': Cls.model.state_dict()}, out_dir + 'best_model.pth')
            print(f'Epoch: {best_epoch}. Save model!')

        # =========================================================================================

        # retrieval
        retrieval_list = []

        embed_feature = netE(dataset.test_unseen_feature.cuda(), emb=True)
        cls_centroid = np.zeros((dataset.ntest_class, opt.resSize))
        for i, class_idx in enumerate(dataset.unseenclasses):
            cls_centroid[i] = torch.mean(syn_feature[syn_label == class_idx,], dim=0)

        dist = cosine_similarity(cls_centroid, embed_feature.cpu())

        precision_100 = torch.zeros(dataset.ntest_class)
        precision_50 = torch.zeros(dataset.ntest_class)
        precision_25 = torch.zeros(dataset.ntest_class)

        dist = torch.from_numpy(-dist)
        for i, class_idx in enumerate(dataset.unseenclasses):
            is_class = dataset.test_unseen_label == class_idx
            cls_num = int(is_class.sum())
            # print(f'class_idx: {class_idx}, cls_num: {cls_num}')

            # 100%
            value, idx = torch.topk(dist[i, :], cls_num, largest=False)
            precision_100[i] = (is_class[idx]).sum().float() / cls_num

            true_retrieval_idx = idx[is_class[idx] == 1]
            false_retrieval_idx = idx[is_class[idx] == 0]

            retrieval_dict = {
                'class_idx': class_idx,
                'top5_true_image': true_retrieval_idx[:5],
                'top5_true_image_name': dataset.test_unseen_image_file[true_retrieval_idx[:5]],
                'top5_false_image': false_retrieval_idx[-5:],
                'top5_false_image_name': dataset.test_unseen_image_file[false_retrieval_idx[-5:]],
            }
            retrieval_list.append(retrieval_dict)

            # 50%
            cls_num_50 = int(cls_num / 2)
            _, idx = torch.topk(dist[i, :], cls_num_50, largest=False)
            precision_50[i] = (is_class[idx]).sum().float() / cls_num_50

            # 25%
            cls_num_25 = int(cls_num / 4)
            _, idx = torch.topk(dist[i, :], cls_num_25, largest=False)
            precision_25[i] = (is_class[idx]).sum().float() / cls_num_25

        if precision_100.mean().item() > best_100_mAP:
            best_100_mAP = precision_100.mean().item()
            best_50_mAP = precision_50.mean().item()
            best_25_mAP = precision_25.mean().item()
            best_retrieval_epoch = epoch
            best_retrieval_list[:] = retrieval_list[:]

        # =========================================================================================

        print(f'dataset: {opt.dataset}, seed: {opt.manualSeed}')
        print(f'unseen:{Cls.unseen_acc}, seen:{Cls.seen_acc}, h:{Cls.h}')
        print(f'Best epoch:{best_epoch}, best unseen:{best_unseen_acc}, best seen:{best_seen_acc}, best h:{best_h}')
        print("retrieval results: 100%%: %.3f, 50%%: %.3f, 25%%: %.3f" % (
            precision_100.mean().item(), precision_50.mean().item(), precision_25.mean().item()))
        print("Best retrieval epoch: %d, best retrieval results 100%%: %.3f, 50%%: %.3f, 25%%: %.3f" % (
            best_retrieval_epoch, best_100_mAP, best_50_mAP, best_25_mAP))
        if opt.dataset == 'AWA2':
            print(f'top-5 true/false images of best retrieval:\n {best_retrieval_list}')
        with open(f"log/log_{opt.dataset}_{start_time}.txt",encoding="utf-8",mode="a+") as f:
            f.write('EP[%d/%d]******************************************************\n' % (epoch, opt.epoch))
            f.write(f'dataset: {opt.dataset}, seed: {opt.manualSeed}\n')
            f.write(f'unseen:{Cls.unseen_acc}, seen:{Cls.seen_acc}, h:{Cls.h}\n')
            f.write(f'Best epoch:{best_epoch}, best unseen:{best_unseen_acc}, best seen:{best_seen_acc}, best h:{best_h}')
            f.write("retrieval results: 100%%: %.3f, 50%%: %.3f, 25%%: %.3f \n" % (
                precision_100.mean().item(), precision_50.mean().item(), precision_25.mean().item()))
            f.write("Best retrieval epoch: %d, best retrieval results 100%%: %.3f, 50%%: %.3f, 25%%: %.3f \n" % (
                best_retrieval_epoch, best_100_mAP, best_50_mAP, best_25_mAP))
            if opt.dataset == 'AWA2':
                f.write(f'top-5 true/false images of best retrieval:\n {best_retrieval_list}\n')

    time_elapsed = time.time() - since
    print('Time Elapsed: {}'.format(time_elapsed))

    netG.train()
    sys.stdout.flush()

time_elapsed = time.time() - since
print('End run!!!')
print('Time Elapsed: {}'.format(time_elapsed))
print(GetNowTime())

