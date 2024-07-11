import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import utils
from tqdm import tqdm
import random
import torch.nn.functional as F
import copy


def rand_no_repeat(low, height, size):
    out = set()
    while len(out) != size:
        i = random.randint(low, height-1)
        out.add(i)

    return torch.LongTensor(list(out))

def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = labels.cuda()
    outputs = torch.log_softmax(logits / T, dim=1).cuda()  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)
    return outputs.cuda()

class Classifier:
    def __init__(self, syn_feature, syn_label, dataset, netE, opt, cur_epoch, generalized = True):
        emb_X = netE(dataset.train_feature.cuda(), emb=True)
        self.emb_X = emb_X
        self.netE = netE
        self.train_X = torch.cat((emb_X.data.cpu(), syn_feature), dim=0)
        test_local = utils.map_label(syn_label, dataset.unseenclasses)
        test_local_label = test_local + dataset.seenclass_num
        self.train_Y = torch.cat((dataset.train_local_label, test_local_label), 0)
        self.S_train_X = self.train_X[:dataset.ntrain]
        self.S_train_Y = dataset.train_local_label
        self.U_train_X = self.train_X[dataset.ntrain:]
        self.U_train_Y = test_local
        self.cur_epoch = cur_epoch
        self.test_unseen_feature = dataset.test_unseen_feature
        self.test_seen_feature = dataset.test_seen_feature
        self.test_unseen_label = dataset.test_unseen_local_label + dataset.seenclass_num
        self.test_seen_label = dataset.test_seen_local_label

        self.seenclasses = dataset.seenclasses
        self.unseenclasses = dataset.unseenclasses

        if opt.dataset == 'AWA1' or opt.dataset == 'AWA2':  # weight for bias loss
            self.ce = 1.0 # 0.9
        else:
            self.ce = opt.ce

        self.opt = opt
        self.nepoch = opt.c_epoch
        self.batch_size = opt.c_batch_size
        self.ntrain = self.train_X.shape[0]
        self.Strain = self.S_train_X.shape[0]
        self.Utrain = self.U_train_X.shape[0]
        self.dataset = dataset
        self.input_dim = opt.hidden_size
        self.train_data = data.TensorDataset(self.train_X, self.train_Y)

        if opt.cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.input = torch.FloatTensor(self.batch_size, self.input_dim).to(self.device)
        self.label = torch.LongTensor(self.batch_size)

        self.model = SoftmaxClassifier(self.input_dim, opt.class_num).to(self.device)
        #self.modelS = SoftmaxClassifier(opt.resSize, opt.seen_num).to(self.device)
        self.modelU = SoftmaxClassifier(self.input_dim, opt.unseen_num).to(self.device)
        self.modelS = SoftmaxClassifier(self.input_dim, opt.seen_num).to(self.device)
        self.model.apply(utils.weights_init)
        #self.modelS.apply(utils.weights_init)
        self.modelU.apply(utils.weights_init)
        self.modelS.apply(utils.weights_init)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        self.optimizerU = optim.Adam(self.modelU.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        self.optimizerS = optim.Adam(self.modelS.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        #self.optimizerS = optim.Adam(self.modelS.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        self.Ubest = copy.deepcopy(self.modelU)
        self.Sbest = copy.deepcopy(self.modelS)
        if generalized:
            self.seen_acc, self.unseen_acc, self.h, self.epoch = self.fit()
        else:
            self.acc, self.epoch = self.fit_zsl()

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        best_epoch = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.Utrain, self.batch_size):
                self.modelU.zero_grad()
                batch_input, batch_label = self.next_batch(self.U_train_X, self.U_train_Y, self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                output = self.modelU(self.input)
                loss = self.criterion(output, self.label.cuda())
                mean_loss += loss.data
                loss.backward()
                self.optimizerU.step()
            pred_unseen = self.valU(self.test_unseen_feature)
            acc = torch.sum(pred_unseen == self.dataset.test_unseen_local_label).float() / \
                         float(self.test_unseen_label.size(0))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
        #print('Training classifier loss= %.4f' % (loss))
        return best_acc, best_epoch

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_epoch = 0
        # distill from unseen_classifier
        if self.cur_epoch >= self.opt.Ustart:
            best_u = 0
            for epoch in range(self.opt.Uepoch):
                for i in range(0, self.Utrain, self.opt.c_batch_size):
                    self.modelU.zero_grad()
                    batch_input, batch_label = self.next_batch(self.U_train_X, self.U_train_Y,self.opt.c_batch_size)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)
                    output = self.modelU(self.input)
                    loss = self.criterion(output, self.label.cuda())
                    loss.backward()
                    self.optimizerU.step()
                pred_unseen = self.valU(self.test_unseen_feature)
                acc_unseen = torch.sum(pred_unseen == self.dataset.test_unseen_local_label).float() / \
                             float(self.dataset.test_unseen_local_label.size(0))
                if acc_unseen>best_u:
                    best_u = acc_unseen
                    self.Ubest = copy.deepcopy(self.modelU)
                    print("Epoch{0}".format(epoch)+":best_unseen:{0}".format(best_u))
            self.Ubest.eval()
            best_s = 0
            for epoch in range(self.opt.Sepoch):
                for i in range(0, self.Strain, self.opt.c_batch_size):
                    self.modelS.zero_grad()
                    batch_input, batch_label = self.next_batch(self.S_train_X, self.S_train_Y,self.opt.c_batch_size)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)
                    output = self.modelS(self.input)
                    loss = self.criterion(output, self.label.cuda())
                    loss.backward()
                    self.optimizerS.step()
                pred_seen = self.valS(self.test_seen_feature)
                acc_seen = torch.sum(pred_seen == self.dataset.test_seen_local_label).float() / \
                             float(self.dataset.test_seen_local_label.size(0))
                if acc_seen>best_s:
                    best_s = acc_seen
                    self.Sbest = copy.deepcopy(self.modelS)
                    print("Epoch{0}".format(epoch)+":best_seen:{0}".format(best_s))
            self.Ubest.eval()
            self.Sbest.eval()

        # print(self.emb_X.shape[0])  # num of train set
        # print(self.test_seen_feature.shape[0])  # num of test seen set
        # print(self.test_unseen_feature.shape[0])  # num of test unseen set
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):  # self.ntrain=22057, self.batch_size=300
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.train_X, self.train_Y, self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                unseen = self.label >= self.opt.seen_num
                unseen_index = unseen.nonzero().squeeze(dim=1)

                seen = self.label < self.opt.seen_num
                seen_index = seen.nonzero().squeeze(dim=1)

                unseen_embed = self.input[unseen_index]
                seen_embed = self.input[seen_index]

                output = self.model(self.input)

                dist_target_u = self.Ubest(unseen_embed)
                dist_target_s = self.Sbest(seen_embed)
                logits_dist_u = output[unseen_index, self.opt.seen_num:]
                logits_dist_s = output[seen_index, :self.opt.seen_num]
                dist_loss_u = MultiClassCrossEntropy(logits_dist_u, dist_target_u, self.opt.class_temp_u)
                dist_loss_s = MultiClassCrossEntropy(logits_dist_s, dist_target_s, self.opt.class_temp_s)
                if self.cur_epoch>=self.opt.Ustart:
                    print("??")
                    loss = self.criterion(output, self.label.cuda()) + self.opt.distill_weight_u * dist_loss_u + self.opt.distill_weight_s * dist_loss_s
                    # loss = loss = bias_loss(output, self.label.cuda(), self.seenclasses, self.unseenclasses, self.ce) + \
                    #               self.opt.distill_weight_u * dist_loss_u + self.opt.distill_weight_s * dist_loss_s
                else:
                    # loss = self.criterion(output, self.label.cuda())
                    loss = bias_loss(output, self.label.cuda(), self.seenclasses, self.unseenclasses, self.ce)
                loss.backward()
                self.optimizer.step()

            pred_unseen = self.val(self.test_unseen_feature)
            acc_unseen = torch.sum(pred_unseen == self.test_unseen_label).float() / \
                         float(self.test_unseen_label.size(0))
            pred_seen = self.val(self.test_seen_feature)
            acc_seen = torch.sum(pred_seen == self.test_seen_label).float() / \
                       float(self.test_seen_label.size(0))
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_H =  round(H.item(), 3)
                best_seen = round(acc_seen.item(), 3)
                best_unseen = round(acc_unseen.item(), 3)
                best_epoch = epoch

        return best_seen, best_unseen, best_H, best_epoch

    def valU(self, test_X):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_X.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            embed = self.netE(test_X[start:end].to(self.device), emb = True)
            output = self.modelU(embed)
            output = output.cpu()

            predicted_label[start:end] = torch.max(output, 1)[1]
            start = end
        return predicted_label
    def valS(self, test_X):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_X.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            embed = self.netE(test_X[start:end].to(self.device), emb = True)
            output = self.modelS(embed)
            output = output.cpu()

            predicted_label[start:end] = torch.max(output, 1)[1]
            start = end
        return predicted_label
    def val(self, test_X):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_X.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            embed = self.netE(test_X[start:end].to(self.device), emb = True)
            output = self.model(embed)
            output = output.cpu()

            predicted_label[start:end] = torch.max(output, 1)[1]
            start = end
        return predicted_label
    # def next_batch(self, train_X, train_Y, batch_size):
    #     # idx = torch.randint(0, self.train_X.shape[0], (batch_size,))
    #     idx = rand_no_repeat(0, train_X.shape[0], batch_size)
    #     batch_feature = train_X[idx]
    #     batch_label = train_Y[idx]
    #     return batch_feature, batch_label
    def next_batch(self, train_X, train_Y, batch_size):
        idx = torch.randperm(train_X.shape[0])[0:batch_size]
        batch_feature = train_X[idx]
        batch_label = train_Y[idx]
        return batch_feature, batch_label


class Classifier_RF:
    def __init__(self, netD, syn_att, syn_feature, syn_label, dataset, netE, opt, cur_epoch, generalized = True):
        emb_X = netE(dataset.train_feature.cuda(), emb=True)
        self.emb_X = emb_X
        self.netE = netE
        self.netD = netD
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = False
        self.repeat_att = utils.map_att(syn_att, dataset.train_local_label, dataset.seenclasses, opt)
        self.train_att = torch.cat((self.repeat_att, syn_att), dim=0)
        self.train_X = torch.cat((emb_X.data.cpu(), syn_feature), dim=0)
        test_local = utils.map_label(syn_label, dataset.unseenclasses)
        test_local_label = test_local + dataset.seenclass_num
        self.train_Y = torch.cat((dataset.train_local_label, test_local_label), 0)
        self.S_train_X = self.train_X[:dataset.ntrain]
        self.S_train_Y = dataset.train_local_label
        self.U_train_X = self.train_X[dataset.ntrain:]
        self.U_train_Y = test_local
        self.cur_epoch = cur_epoch
        self.test_unseen_feature = dataset.test_unseen_feature
        self.test_seen_feature = dataset.test_seen_feature
        self.test_unseen_label = dataset.test_unseen_local_label + dataset.seenclass_num
        self.test_seen_label = dataset.test_seen_local_label

        self.seenclasses = dataset.seenclasses
        self.unseenclasses = dataset.unseenclasses

        if opt.dataset == 'AWA1' or opt.dataset == 'AWA2':  # weight for bias loss
            self.ce = 0.9
        else:
            self.ce = opt.ce

        self.opt = opt
        self.nepoch = opt.c_epoch
        self.batch_size = opt.c_batch_size
        self.ntrain = self.train_X.shape[0]
        self.Strain = self.S_train_X.shape[0]
        self.Utrain = self.U_train_X.shape[0]
        self.dataset = dataset
        self.input_dim = opt.hidden_size
        self.train_data = data.TensorDataset(self.train_X, self.train_Y)

        if opt.cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.input = torch.FloatTensor(self.batch_size, self.input_dim).to(self.device)
        self.label = torch.LongTensor(self.batch_size)
        self.att = torch.FloatTensor(self.batch_size, dataset.att_dim).to(self.device)

        self.model = SoftmaxClassifier(self.input_dim, opt.class_num).to(self.device)
        #self.modelS = SoftmaxClassifier(opt.resSize, opt.seen_num).to(self.device)
        self.modelU = SoftmaxClassifier(self.input_dim, opt.unseen_num).to(self.device)
        self.modelS = SoftmaxClassifier(self.input_dim, opt.seen_num).to(self.device)
        self.model.apply(utils.weights_init)
        #self.modelS.apply(utils.weights_init)
        self.modelU.apply(utils.weights_init)
        self.modelS.apply(utils.weights_init)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        self.optimizerU = optim.Adam(self.modelU.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        self.optimizerS = optim.Adam(self.modelS.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        #self.optimizerS = optim.Adam(self.modelS.parameters(), lr=opt.c_lr, betas=(0.5, 0.999))
        self.Ubest = copy.deepcopy(self.modelU)
        self.Sbest = copy.deepcopy(self.modelS)
        if generalized:
            self.seen_acc, self.unseen_acc, self.h, self.epoch = self.fit()
        else:
            self.acc, self.epoch = self.fit_zsl()

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        best_epoch = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.Utrain, self.batch_size):
                self.modelU.zero_grad()
                batch_input, batch_label = self.next_batch(self.U_train_X, self.U_train_Y, self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                output = self.modelU(self.input)
                loss = self.criterion(output, self.label.cuda())
                mean_loss += loss.data
                loss.backward()
                self.optimizerU.step()
            pred_unseen = self.valU(self.test_unseen_feature)
            acc = torch.sum(pred_unseen == self.dataset.test_unseen_local_label).float() / \
                         float(self.test_unseen_label.size(0))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
        #print('Training classifier loss= %.4f' % (loss))
        return best_acc, best_epoch

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_epoch = 0
        # distill from unseen_classifier
        if self.cur_epoch >= self.opt.Ustart:
            best_u = 0
            for epoch in range(self.opt.Uepoch):
                for i in range(0, self.Utrain, self.opt.c_batch_size):
                    self.modelU.zero_grad()
                    batch_input, batch_label = self.next_batch(self.U_train_X, self.U_train_Y,self.opt.c_batch_size)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)
                    output = self.modelU(self.input)
                    loss = self.criterion(output, self.label.cuda())
                    loss.backward()
                    self.optimizerU.step()
                pred_unseen = self.valU(self.test_unseen_feature)
                acc_unseen = torch.sum(pred_unseen == self.dataset.test_unseen_local_label).float() / \
                             float(self.dataset.test_unseen_local_label.size(0))
                if acc_unseen>best_u:
                    best_u = acc_unseen
                    self.Ubest = copy.deepcopy(self.modelU)
                    print("Epoch{0}".format(epoch)+":best_unseen:{0}".format(best_u))
            self.Ubest.eval()
            best_s = 0
            for epoch in range(self.opt.Sepoch):
                for i in range(0, self.Strain, self.opt.c_batch_size):
                    self.modelS.zero_grad()
                    batch_input, batch_label = self.next_batch(self.S_train_X, self.S_train_Y,self.opt.c_batch_size)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)
                    output = self.modelS(self.input)
                    loss = self.criterion(output, self.label.cuda())
                    loss.backward()
                    self.optimizerS.step()
                pred_seen = self.valS(self.test_seen_feature)
                acc_seen = torch.sum(pred_seen == self.dataset.test_seen_local_label).float() / \
                             float(self.dataset.test_seen_local_label.size(0))
                if acc_seen>best_s:
                    best_s = acc_seen
                    self.Sbest = copy.deepcopy(self.modelS)
                    print("Epoch{0}".format(epoch)+":best_seen:{0}".format(best_s))
            self.Ubest.eval()
            self.Sbest.eval()

        # print(self.emb_X.shape[0])  # num of train set
        # print(self.test_seen_feature.shape[0])  # num of test seen set
        # print(self.test_unseen_feature.shape[0])  # num of test unseen set
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):  # self.ntrain=22057, self.batch_size=300
                self.model.zero_grad()
                batch_input, batch_label, batch_att = self.next_batch(self.train_X, self.train_Y, self.train_att, self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                self.att.copy_(batch_att)

                unseen = self.label >= self.opt.seen_num
                unseen_index = unseen.nonzero().squeeze(dim=1)

                seen = self.label < self.opt.seen_num
                seen_index = seen.nonzero().squeeze(dim=1)

                unseen_embed = self.input[unseen_index]
                seen_embed = self.input[seen_index]

                _, mus, _ = self.netD(torch.cat((self.input, self.att), dim=1))
                output = self.model(mus)

                dist_target_u = self.Ubest(unseen_embed)
                dist_target_s = self.Sbest(seen_embed)
                logits_dist_u = output[unseen_index, self.opt.seen_num:]
                logits_dist_s = output[seen_index, :self.opt.seen_num]
                dist_loss_u = MultiClassCrossEntropy(logits_dist_u, dist_target_u, self.opt.class_temp_u)
                dist_loss_s = MultiClassCrossEntropy(logits_dist_s, dist_target_s, self.opt.class_temp_s)
                if self.cur_epoch>=self.opt.Ustart:
                    print("??")
                    loss = self.criterion(output, self.label.cuda()) + self.opt.distill_weight_u * dist_loss_u + self.opt.distill_weight_s * dist_loss_s
                    # loss = loss = bias_loss(output, self.label.cuda(), self.seenclasses, self.unseenclasses, self.ce) + \
                    #               self.opt.distill_weight_u * dist_loss_u + self.opt.distill_weight_s * dist_loss_s
                else:
                    # loss = self.criterion(output, self.label.cuda())
                    loss = bias_loss(output, self.label.cuda(), self.seenclasses, self.unseenclasses, self.ce)
                loss.backward()
                self.optimizer.step()

            pred_unseen = self.val(self.test_unseen_feature)
            acc_unseen = torch.sum(pred_unseen == self.test_unseen_label).float() / \
                         float(self.test_unseen_label.size(0))
            pred_seen = self.val(self.test_seen_feature)
            acc_seen = torch.sum(pred_seen == self.test_seen_label).float() / \
                       float(self.test_seen_label.size(0))
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_H =  round(H.item(), 3)
                best_seen = round(acc_seen.item(), 3)
                best_unseen = round(acc_unseen.item(), 3)
                best_epoch = epoch

        return best_seen, best_unseen, best_H, best_epoch

    def valU(self, test_X):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_X.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            embed = self.netE(test_X[start:end].to(self.device), emb = True)
            output = self.modelU(embed)
            output = output.cpu()

            predicted_label[start:end] = torch.max(output, 1)[1]
            start = end
        return predicted_label
    def valS(self, test_X):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_X.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            embed = self.netE(test_X[start:end].to(self.device), emb = True)
            output = self.modelS(embed)
            output = output.cpu()

            predicted_label[start:end] = torch.max(output, 1)[1]
            start = end
        return predicted_label
    def val(self, test_X):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_X.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            embed = self.netE(test_X[start:end].to(self.device), emb = True)
            output = self.model(embed)
            output = output.cpu()

            predicted_label[start:end] = torch.max(output, 1)[1]
            start = end
        return predicted_label
    # def next_batch(self, train_X, train_Y, batch_size):
    #     # idx = torch.randint(0, self.train_X.shape[0], (batch_size,))
    #     idx = rand_no_repeat(0, train_X.shape[0], batch_size)
    #     batch_feature = train_X[idx]
    #     batch_label = train_Y[idx]
    #     return batch_feature, batch_label
    def next_batch(self, train_X, train_Y, syn_att, batch_size):
        idx = torch.randperm(train_X.shape[0])[0:batch_size]
        batch_feature = train_X[idx]
        batch_label = train_Y[idx]
        batch_att = syn_att[idx]
        return batch_feature, batch_label, batch_att


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)

    def forward(self, x):
        o = self.fc(x)
        return o

def bias_loss(out, target, seenclasses, unseenclasses, c=1):
    out = torch.exp(out)
    class_num = seenclasses.shape[0] + unseenclasses.shape[0]
    one_hot_label_weight = torch.eye(class_num)[target, :]
    one_hot_label_weight[:, unseenclasses] *= c
    weight = torch.ones_like(one_hot_label_weight)
    weight[:, unseenclasses] *= c
    p = one_hot_label_weight.to(out.device) * out
    loss = -torch.sum(torch.log(torch.sum(p, dim=1) / torch.sum(weight.to(out.device) * out, dim=1)), dim=0) / out.shape[0]
    return loss
