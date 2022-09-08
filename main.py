import json
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import os
from utils import cache_transformed_train_data, cache_transformed_test_data
from utils import transform_label_CT_MRI, train_transforms, adjust_learning_rate, set_requires_grad, MI_loss, plot_fn, set_up_logger, deactivate_batchnorm
import monai
from efficientunet import *
from model.GIN import GIN
import tqdm
import itertools
from model.cut_model import ResnetGenerator, PatchSampleF
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Train the model")

parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")
parser.add_argument("--train_dataset", type=str, default="Abdominal_CT", help='name of training dataset')
parser.add_argument("--test_dataset", type=str, default="CHAOS_MRI", help='name of training dataset')
parser.add_argument("--train_batch_size", type=int, default=1, help="batch size of the forward pass")
parser.add_argument("--epochs", type=int, default=1000, help="num of epochs")
parser.add_argument("--img_size", type=int, default=192, help="image size")
parser.add_argument("--num_classes", type=int, default=5, help="num of classes")
parser.add_argument("--lr", type=float, default=0.0003, help="batch size of the forward pass")


parser.add_argument(
            "--GIN", action='store_true', help='global intensity non-linear augmentation'
        )
########## MI parameters
parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

parser.add_argument(
            "--MSP", action='store_true', help='global intensity non-linear augmentation'
        )
parser.add_argument(
            "--adv", action='store_true', help='global intensity non-linear augmentation'
        )

parser.add_argument("--num_workers", type=int, default=12, help="num of data loading workers")
parser.add_argument("--cache_rate", type=float, default=1.0, help="batch size of the forward pass")
parser.set_defaults(hardness=True)
parser.add_argument(
            "--results_folder_name", type=str, default="../results", help="name of results folder"
        )
parser.add_argument(
            "--data_root", type=str, default= '', help="name of results folder"
        )
parser.add_argument('--world_size', type=int, default=4, help='world size of distrbuted learning')
parser.add_argument('--rank', type=int, default=0, help='rank of distrbuted learning')
parser.add_argument('--master_port', type=str, default='69280', help='rank of distrbuted learning')

args = parser.parse_args()

args.results_folder_name = os.path.join(args.results_folder_name, ('EMP_' if not args.GIN else 'GIN_')
                                        + ('MSP' if args.MSP else '') + ('adv' if args.adv else ''))
if not os.path.exists(args.results_folder_name):
    os.mkdir(args.results_folder_name)

logger = set_up_logger(args.results_folder_name, 'log.txt')

args.nce_layers = [int(i) for i in args.nce_layers.split(',')]
#######training loader
train_file = open(args.train_dataset, 'r')
train_lists = json.load(train_file)
l = len(train_lists)

train_split = []
eval_split = []

test = np.random.choice(l, int(l*0.3), replace=False)

for i in range(l):
    if i in test:
        eval_split.append(train_lists[i])
    else:
        train_split.append(train_lists[i])

train_transform = train_transforms[args.train_dataset]
train_eval_transform = train_transforms[args.train_dataset+'_eval']
train_loader = cache_transformed_train_data(args, train_split, train_transform)
train_eval_loader = cache_transformed_test_data(args, eval_split, train_eval_transform)

######testing loader
testing_file = open(args.test_dataset, 'r')
test_lists = json.load(testing_file)
print(len(test_lists))
test_transform = train_transforms[args.test_dataset]
test_loader = cache_transformed_test_data(args, test_lists, test_transform)

model = get_efficientunet_b2(out_channels=args.num_classes, concat_input=True, pretrained=True).cuda()
# model.apply(deactivate_batchnorm)
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         m.affine = False
GIN = GIN(args).cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
# opt = torch.optim.SGD(model.parameters(), lr=args.lr, betas=(0.5, 0.999), momentum=0.9, weight_decay=5e-6)

if args.adv:
    netG = ResnetGenerator(input_nc=3, output_nc=3, ngf=8, norm_layer=nn.InstanceNorm2d,
                          use_dropout=False, no_antialias=False, no_antialias_up=False, n_blocks=6, ).cuda()

    netF = PatchSampleF(use_mlp=True, init_gain=0.02).cuda()

    opt_gin = torch.optim.Adam(GIN.parameters(), lr=args.lr, betas=(0.5, 0.999))
    flag_MI = True
else:
    set_requires_grad(GIN, False)

Loss_Fn = monai.losses.DiceCELoss(softmax=True, to_onehot_y=True)#monai.losses.DiceLoss(softmax=True, to_onehot_y=True)#

eval_Fn = monai.losses.DiceLoss(softmax=True, to_onehot_y=True)
eval_metric = monai.metrics.DiceMetric(reduction="mean_batch")

best_eval = -1
best_epoch = -1
best_test = 0
best_test_score = []
pbar = tqdm.tqdm(range(args.epochs))
for epoch in pbar:
    adjust_learning_rate(opt,epoch,args.lr,args.epochs)
    if args.GIN and args.adv:
        adjust_learning_rate(opt_gin,epoch,args.lr,args.epochs)
        if not flag_MI:
            adjust_learning_rate(opt_MI,epoch,args.lr,args.epochs)
    model.train()
    epoch = epoch+1

    pbar_b = tqdm.tqdm(enumerate(train_loader))
    for i, batch_data in pbar_b:
        img, label = batch_data['image'], batch_data['label']

        img = img.float().cuda().permute(4, 0, 1, 2, 3).reshape(-1,1,192,192)
        label = label.long().cuda().permute(4, 0, 1, 2, 3).reshape(-1,1,192,192)
        label = transform_label_CT_MRI(label)

        img = torch.cat([img] * 3, dim=1)

        loss = torch.zeros(1).cuda()
        loss_aug1 = torch.zeros(1).cuda()
        loss_aug2 = torch.zeros(1).cuda()
        loss_kl = torch.zeros(1).cuda()
        loss_MI = torch.zeros(1).cuda()

        opt.zero_grad()

        if args.GIN and args.adv:
            opt_gin.zero_grad()
            if not flag_MI:
                opt_MI.zero_grad()

        if not args.GIN:

            predict = model(img)
            loss = Loss_Fn(predict, label)
            loss.backward()

        else:
            if args.adv:
                aug_img1, aug_img2 = GIN(img)
                if np.random.choice(2,1) == 1:
                    loss_MI = MI_loss(aug_img1, aug_img2, [netG, netF], args)
                else:
                    loss_MI = MI_loss(aug_img2, aug_img1, [netG, netF], args)
                if flag_MI:
                    print('init F')
                    netF = netF.cuda()
                    opt_MI = torch.optim.Adam(itertools.chain(netG.parameters(), netF.parameters()), lr=args.lr, betas=(0.5, 0.999))
                    opt_MI.zero_grad()
                    flag_MI = False
                    loss_MI = MI_loss(aug_img1, aug_img2, [netG, netF], args)
            else:
                with torch.no_grad():
                    aug_img1, aug_img2 = GIN(img)

            predict1 = model(aug_img1.detach())
            predict2 = model(aug_img2.detach())
            loss_aug1 = Loss_Fn(predict1, label)
            # loss_aug1.backward()
            loss_aug2 = Loss_Fn(predict2, label)
            # loss_aug2.backward()
            prob1 = torch.softmax(predict1, dim=1)
            prob2 = torch.softmax(predict2, dim=1)
            loss_kl = (nn.KLDivLoss()(prob1, prob2) + nn.KLDivLoss()(prob2, prob1))/2

            (loss_kl * 10.0 + loss_aug1 + loss_aug2).backward()
            opt.step()

            if args.GIN and args.adv:
                set_requires_grad(model, False)
                predict1 = model(aug_img1)
                predict2 = model(aug_img2)
                prob1 = torch.softmax(predict1, dim=1)
                prob2 = torch.softmax(predict2, dim=1)
                loss_kl = (nn.KLDivLoss()(prob1, prob2) + nn.KLDivLoss()(prob2, prob1))/2
                (-loss_kl * 10.0 + loss_MI*10).backward()
                set_requires_grad(model, True)

        opt.step()
        if args.GIN and args.adv:
            opt_gin.step()
            opt_MI.step()

        pbar_b.set_description('loss_{0:.2f}, lossaug1_{1:.2f}, lossaug2_{2:.2f}, losskl_{3:.2f}, lossMI_{4:.2f}'.format(loss.item(), loss_aug1.item(), loss_aug2.item(), loss_kl.item(), loss_MI.item()))

    # plot_fn(img, label)
    if epoch % 100 == 0 and epoch!= 0:
        torch.save(model.state_dict(), os.path.join(args.results_folder_name, 'epoch_{:d}'.format(epoch)))

    if epoch % 5 == 0 and epoch != 0:
        with torch.no_grad():
            model.eval()

            eval_dice_score = []
            eval_list = []
            for i, batch_data in enumerate(train_eval_loader):
                img, label = batch_data['image'], batch_data['label']

                img = img.float().cuda().permute(4, 0, 1, 2, 3).squeeze(dim=1)
                label = label.long().cuda().permute(4, 0, 1, 2, 3).squeeze(dim=1)
                label = transform_label_CT_MRI(label)
                img = torch.cat([img] * 3, dim=1)
                predict = model(img)

                loss = eval_Fn(predict, label)
                eval_list.append(loss.item())
                predict_label = torch.argmax(predict, dim=1).squeeze()

                label_onehot = label.squeeze()

                predict_label = F.one_hot(predict_label, num_classes=5).permute(3, 1, 2, 0).unsqueeze(dim=0)
                label_onehot = F.one_hot(label_onehot, num_classes=5).permute(3, 1, 2, 0).unsqueeze(dim=0)

                eval_dice_score.append(eval_metric(predict_label,
                                              label_onehot).detach().cpu().numpy())  # DS_class(predict_label,label,metric=eval_metric,num_classes=args.num_classes)

            eval_dice_score = np.asarray(eval_dice_score)

            logger.info('Eval: ', )
            logger.info(np.ndarray.tolist(eval_dice_score.mean(axis=0)))

            test_dice_score = []
            test_list = []
            for i, batch_data in enumerate(test_loader):
                img, label = batch_data['image'], batch_data['label']

                img = img.float().cuda().permute(4, 0, 1, 2, 3).squeeze(dim=1)
                label = label.long().cuda().permute(4, 0, 1, 2, 3).squeeze(dim=1)
                img = torch.cat([img] * 3, dim=1)
                predict = model(img)

                loss = eval_Fn(predict, label)
                test_list.append(loss.item())
                predict_label = torch.argmax(predict, dim=1).squeeze()

                label_onehot = label.squeeze()

                predict_label = F.one_hot(predict_label, num_classes=5).permute(3, 1, 2, 0).unsqueeze(dim=0)
                label_onehot = F.one_hot(label_onehot, num_classes=5).permute(3, 1, 2, 0).unsqueeze(dim=0)

                test_dice_score.append(eval_metric(predict_label,
                                              label_onehot).detach().cpu().numpy())  # DS_class(predict_label,label,metric=eval_metric,num_classes=args.num_classes)

            test_dice_score = np.asarray(test_dice_score)

            logger.info('Test: ')
            logger.info(test_dice_score.mean(axis=0))

            if best_eval < eval_dice_score.mean():
                best_eval = eval_dice_score.mean()
                best_epoch = epoch
                best_test_score = test_dice_score.mean(axis=0)
                torch.save(model.state_dict(), os.path.join(args.results_folder_name, 'best'))
                torch.save(GIN.state_dict(), os.path.join(args.results_folder_name, 'GIN'))

            logger.info('Best_Test: ')
            logger.info(np.ndarray.tolist(best_test_score))
            logger.info(' at epoch ')
            logger.info( str(best_epoch))
            # # plot_fn(img, label)
