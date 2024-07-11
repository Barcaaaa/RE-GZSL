import argparse


class Options(object):
    def __init__(self):
        # Training settings

        parser = argparse.ArgumentParser(description='Tank Shot')

        # data parameter
        parser.add_argument('--dataset', default='AWA1')

        parser.add_argument('--dataroot', default='/data1/wuyao/dataset/gzsl_data', help='path to dataset')

        parser.add_argument('--image_embedding', default='res101')
        parser.add_argument('--len', type = int, default=50)

        parser.add_argument('--class_embedding', default='att')

        parser.add_argument('--preprocessing', action='store_true', default=True,
                            help='enable MinMaxScaler on visual features')

        parser.add_argument('--standardization', action='store_true', default=False)

        parser.add_argument('--fine_tuning', action='store_true', default=False, help='')

        parser.add_argument('--cuda', action='store_true', default=True)

        parser.add_argument('--resSize', type=int, default=2048, help='size of features')

        parser.add_argument('--embSize', type=int, default=2048, help='size of features')

        parser.add_argument('--attSize', type=int, default=85, help='size of attributes')

        parser.add_argument('--class_num', type=int, default=50, help='num of classes')

        parser.add_argument('--unseen_num', type=int, default=10, help='num of unseen classes')

        parser.add_argument('--seen_num', type=int, default=40, help='num of seen classes')

        parser.add_argument('--noiseSize', type=int, default=85, help='size of noise')

        parser.add_argument('--syn_num', type=int, default=5000, help='number features to generate per class')

        parser.add_argument('--batch_size', type=int, default=512, help='input batch size for generator')

        parser.add_argument('--c_batch_size', type=int, default=1024, help='input batch size for classifier')

        parser.add_argument('--shot', type=int, default=4, help='input batch size for classifier')

        parser.add_argument('--way', type=int, default=3, help='input batch size for classifier')

        parser.add_argument('--epoch', type=int, default=2000, help='number of epochs to train for generator')

        parser.add_argument('--c_epoch', type=int, default=30, help='number of epochs to train for classifier')

        parser.add_argument('--best_epoch', type=int, default=0, help='')

        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train for model')

        parser.add_argument('--c_lr', type=float, default=0.001, help='learning rate to train for classifier')

        parser.add_argument('--ce', type=float, default=1, help='weight of cross entropy')

        parser.add_argument('--use_mome', type=bool, default=True, help='weight of ')

        parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')

        parser.add_argument('--lambda0', type=float, default=10, help='gradient penalty regularize, following WGAN-GP')

        parser.add_argument('--gammaD', type=float, default=1, help='gradient penalty regularize, following WGAN-GP')

        parser.add_argument('--gammaG', type=float, default=1, help='gradient penalty regularize, following WGAN-GP')

        parser.add_argument('--manualSeed', type=int, help='manual seed')

        parser.add_argument('--gzsl', type=bool, default=True, help='gzsl or not')

        parser.add_argument('--cub_att', type=str, default= 'sent', help='sent or att')

        parser.add_argument('--proSize', type=int, default=512, help='size of projection')

        parser.add_argument('--temperature', type=float, default=10, help='temperature in loss_fn')

        parser.add_argument('--gpus', default=0, type=int, help='type of loss')

        parser.add_argument('--iter_per_epoch', default=100, type=int)
        parser.add_argument('--seen_Neighbours', default=10, type=int, help='image size')
        parser.add_argument('--outzSize', default=512, type=int, help='weight decay')
        parser.add_argument('--hidden_size', default=2048, type=int, help='weight decay')
        parser.add_argument('--dist_ratio', default=1, type=float)
        parser.add_argument('--angle_ratio', default=2, type=float)
        parser.add_argument('--align_ratio', default=0.001, type=float)
        parser.add_argument('--embed_ratio', default=1, type=float)
        parser.add_argument('--ce_ratio', default=1, type=float)
        parser.add_argument('--distill_ratio', default=3, type=float)
        parser.add_argument('--gan_weight', default=1, type=float)
        parser.add_argument('--E_weight', default=0.001, type=float)
        parser.add_argument('--gamma', default=0.9, type=float)
        parser.add_argument('--beta', default=1e-3, type=float, help='beta')
        parser.add_argument('--contrast_ratio', default=0.1, type=float, help='beta')
        parser.add_argument('--mad', default=0.5, type=float, help='weight decay')

        parser.add_argument('--Ustart', default=20000, type=int, help='weight decay')
        parser.add_argument('--Sstart', default=20000, type=int, help='weight decay')
        parser.add_argument('--class_temp_u', default=1.0, type=float, help='weight decay')
        parser.add_argument('--class_temp_s', default=1.0, type=float, help='weight decay')
        parser.add_argument('--Ubatch_size', type=int, default=512, help='input batch size for classifier')
        parser.add_argument('--Sbatch_size', type=int, default=512, help='input batch size for classifier')
        parser.add_argument('--Uepoch', type=int, default=70, help='number of epochs to train for classifier')
        parser.add_argument('--Sepoch', type=int, default=70, help='number of epochs to train for classifier')

        parser.add_argument('--without_FM', action='store_true', default=False)        

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
