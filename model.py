import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize+opt.attSize, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        o = self.fc2(h)
        return o


class Discriminator_RF1(nn.Module):
    def __init__(self, opt):
        super(Discriminator_RF1, self).__init__()
        self.latenSize = 2048
        self.fc1 = nn.Linear(opt.resSize+opt.attSize, 2 * self.latenSize)  # mapping func
        self.sigmoid = nn.Sigmoid()

        self.fc2 = nn.Linear(self.latenSize, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.apply(weights_init)

    def reparameter(self, mu, sigma):
        return (torch.randn_like(mu) * sigma) + mu

    def forward(self, x, train_G=False):
        h = self.lrelu(self.fc1(x))
        mus, stds = h[:, :self.latenSize], h[:, self.latenSize:]
        stds = self.sigmoid(stds)
        encoder_out = self.reparameter(mus, stds)
        if not train_G:
            o = self.fc2(encoder_out)
        else:
            o = self.fc2(mus)
        return o, mus, stds


class Embedding_model(nn.Module):
    def __init__(self, opt, dataset):
        super(Embedding_model, self).__init__()

        self.opt = opt
        self.left = nn.Sequential(
            nn.Linear(opt.resSize, opt.hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(opt.hidden_size, opt.outzSize),
            nn.Linear(opt.outzSize, opt.class_num))

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.fc1 = nn.Linear(opt.resSize, opt.hidden_size)
        self.fc2 = nn.Linear(opt.hidden_size, opt.outzSize)
        self.fc3 = nn.Linear(opt.outzSize, opt.class_num)

        self.relu = nn.ReLU(True)
        self.momentum = opt.mad
        self.dataset = dataset
        self.get_center()
        self.apply(weights_init)
        self.criterion = nn.CrossEntropyLoss()

    def get_center(self):
        center_emb = self.lrelu(self.fc1(self.dataset.tr_cls_centroid))
        self.center = self.fc2(center_emb)
        self.center = self.center.detach().cuda()

    def update_center(self, att, feature, label):
        att_weight = att.mm(att.t())
        new_feature = torch.matmul(att_weight, feature)
        unique_class = torch.unique(label)
        for i in unique_class:
            select_feature = new_feature[label == i]
            mean_feature = torch.mean(select_feature, dim=0)
            self.center[i] = self.momentum * self.center[i].detach() + (1. - self.momentum) * mean_feature

    def forward(self, features, label=None, local_label=None, emb=False, retrieval=False):
        left_logits = self.left(features)
        embedding = self.lrelu(self.fc1(features))
        out_z = self.fc2(embedding)
        right_logits = self.fc3(out_z)
        out_z_norm = F.normalize(out_z, p=2, dim=1)
        if emb == True:
            return embedding
        if retrieval == True:
            return embedding, out_z

        kd_loss = MultiClassCrossEntropy(right_logits, left_logits.detach())
        ce_left = self.criterion(left_logits, label).mean()
        ce_right = self.criterion(right_logits, label).mean()
        loss_cls = self.opt.ce_ratio * (ce_left + ce_right) + self.opt.distill_ratio * kd_loss

        compare_center = self.center
        compare_center_norm = F.normalize(compare_center, p=2, dim=1)
        anchor_dot_contrast = torch.div(torch.matmul(out_z_norm, compare_center_norm.t().detach()),
                            self.opt.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = F.one_hot(local_label, num_classes=self.dataset.seenclass_num).float().cuda()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        all_loss = loss_cls + self.opt.contrast_ratio * loss

        return out_z_norm, out_z, all_loss


class Embedding_model_baseline(nn.Module):
    def __init__(self, opt, dataset):
        super(Embedding_model_baseline, self).__init__()

        self.opt = opt

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.fc1 = nn.Linear(opt.resSize, opt.hidden_size)
        self.fc2 = nn.Linear(opt.hidden_size, opt.outzSize)
        self.fc3 = nn.Linear(opt.outzSize, opt.class_num)

        self.apply(weights_init)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, label = None, local_label = None, emb = False):
        embedding = self.lrelu(self.fc1(features))
        out_z = self.fc2(embedding)
        right_logits = self.fc3(out_z)
        if emb == True:
            return embedding

        loss_cls = self.criterion(right_logits, label).mean()

        return loss_cls


def MultiClassCrossEntropy(logits, labels):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = labels.cuda()
    logits_normed = F.normalize(logits, dim=-1, p=2)
    labels_normed = F.normalize(labels, dim=-1, p=2)
    outputs = torch.log_softmax(logits_normed, dim=1).cuda()  # compute the log of softmax values
    labels = torch.softmax(labels_normed, dim=1)

    outputs_softmax = torch.softmax(logits_normed, dim=1)

    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)

    return outputs.cuda()

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(opt.attSize+opt.noiseSize, 4096)
        self.fc2 = nn.Linear(4096, opt.resSize)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, noise, att):
        i = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(i))
        o = self.relu(self.fc2(h))
        return o

