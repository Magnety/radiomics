from DataOperate import MySet,get_data_list
from Utils import DiceLoss,metrics
from focal_loss import focal_loss
from myVGG import vgg16_bn,vgg11_bn
import time
import os
import numpy as np
import cv2
import SimpleITK as itk
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
from NN.data_aug.data_augmentation_moreDA import get_moreDA_augmentation
from NN.data_aug.default_data_augmentation import default_3D_augmentation_params

def get_k_fold_data(k, i, X):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
    train_list, val_list = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        part = X[idx]
        if j == i:  ###第i折作valid
            val_list = part
        elif train_list is None:
            train_list = part
        else:
            train_list = train_list + part  # dim=0增加行数，竖着连接
    return train_list, val_list
def k_fold(k, X, train_log, val_cls_log,tensorboard_dir):
    for i in range(k):
        writer = SummaryWriter(tensorboard_dir+'/fold%d'%i)
        total = 0
        positive = 0
        torch.cuda.empty_cache()
        net =vgg11_bn()
        net = torch.nn.DataParallel(net, device_ids).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5,weight_decay=0.05)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75)
        train_list, val_lsit = get_k_fold_data(k, i, X)  # 获取k折交叉验证的训练和验证数据
        train_set = MySet(train_list)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_set = MySet(val_lsit)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        data_aug_params = default_3D_augmentation_params
        tr_gen, val_gen = get_moreDA_augmentation(
            train_loader,val_loader,
            (25,79,49),
            data_aug_params,
            deep_supervision_scales=[[1,1,1]],
            pin_memory=True,
            use_nondetMultiThreadedAugmenter=False
        )
        for batch_idx, (image, mask, feature,label, name) in enumerate(tr_gen):
            label = Variable(label[0].cuda())
            l_class = label.data.cpu().numpy()[0]
            if l_class == 1:
                positive += 1
            total += 1
        positive_weight = positive / total
        criterion_mse = focal_loss(alpha=positive_weight)
        ### 每份数据进行训练,体现步骤三####
        train(i, net, tr_gen, val_gen, optimizer, criterion_mse, scheduler,
              train_log, val_cls_log,writer)


def train(k, net, train_loader, val_loader, optimizer,  criterion_mse, scheduler,train_log,val_cls_log,writer):
    for epoch in range(0, 80):
        best_dice = 0.
        best_recall = 0.
        epoch_start_time = time.time()
        print("Epoch: {}".format(epoch))
        epoch_loss = 0.
        total_num = 0
        true_num = 0
        p_num = 0
        p_thresh = 0
        p_min = 1
        for batch_idx, (image,mask,feature,label,name) in enumerate(train_loader):
            print(name)
            start_time = time.time()
            image = Variable(image.cuda())
            label = Variable(label.cuda())
            feature = Variable(feature.cuda())
            cls_out = net(image,feature)
            print(cls_out)
            cls_out = cls_out.to(torch.float32)
            print(cls_out)
            #cls_out = torch.unsqueeze(cls_out, 0)
            print(cls_out)
            cls_loss = criterion_mse(cls_out, label)
            cls_out_cpu = cls_out.data.cpu().numpy()
            print(cls_out_cpu)
            positive_p = np.exp(cls_out_cpu[0][1])
            negtive_p = 1 - positive_p
            print("positive_posibility: %f" % positive_p + "negtive_posibility: %f" % negtive_p)
            print(label)
            output_label = cls_out_cpu.argmax()
            gt_label = label.data.cpu().numpy()[0]
            print("Output label:", output_label)
            print("True label:", gt_label)
            if gt_label == 1 and output_label == 1:
                if p_min > positive_p:
                    p_min = positive_p
                p_num += 1
                p_thresh += positive_p
            if output_label == gt_label:
                true_num += 1
            total_num += 1
            optimizer.zero_grad()

            loss =  cls_loss
            #loss = 1.0 * loss0_dice + 4.0 * cls_loss + 0.5 * loss0_bce

            epoch_loss += loss.item()
            print('Fold: {} | Epoch: {} | Batch: {} | Patient: {:20}----->Train loss: {:4f} | Cost Time: {}\n'.format(k, epoch, batch_idx,name[0],loss.item(),time.time() - start_time))
            open(train_log, 'a').write('Fold: {} | Epoch: {} | Batch: {} | Patient: {:30}----->Train loss: {:4f} | Cost Time: {}\n'.format(k, epoch, batch_idx,name[0],loss.item(),time.time() - start_time))
            loss.backward()
            optimizer.step()
        print("Fold: {} | Epoch: {} | Loss: {} | Acc: {} | Time: {}\n".format(k, epoch, epoch_loss / (batch_idx + 1),true_num/total_num,time.time() - epoch_start_time))
        open(train_log, 'a').write("Fold: {} | Epoch: {} | Loss: {} | Acc: {} | Time: {}\n".format(k, epoch, epoch_loss / (batch_idx + 1),true_num/total_num,time.time() - epoch_start_time))
        writer.add_scalar('train_loss', epoch_loss / (batch_idx + 1), global_step=epoch)
        writer.add_scalar('train_acc', true_num/total_num, global_step=epoch)
        scheduler.step()
        # begin to eval
        net.eval()
        total_num = 0
        true_num = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        true_num1 = 0
        tp1 = 0
        fp1 = 0
        tn1 = 0
        fn1 = 0
        dice = 0.
        jaccard = 0.
        seg_precision = 0.
        seg_sensitivity = 0.
        seg_specificity = 0.
        hd95 = 0.
        with torch.no_grad():
            for batch_idx, (image, mask,feature, label, name) in enumerate(val_loader):
                image = Variable(image.cuda())
                label = Variable(label.cuda())
                feature = Variable(feature.cuda())
                cls_out = net(image,feature)
                # classification metrics
                cls_out = cls_out.to(torch.float32)
                cls_out_cpu = cls_out.data.cpu().numpy()
                gt_label = label.data.cpu().numpy()[0]
                total_num += 1
                positive_p = np.exp(cls_out_cpu[0][1])
                if positive_p >=0.5:
                    output_label = 1
                else:
                    output_label = 0
                if output_label == gt_label:
                    true_num += 1
                    if output_label == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if output_label == 1:
                        fp += 1
                    else:
                        fn += 1

                negtive_p = 1 - positive_p
                open(val_cls_log, 'a').write(
                    "Fold: {} | Epoch: {} | Patient: {:20} | positve_posibility: {:5f} | nagtive_posibility: {:5f} | Gt_Label: {} | Val_Label: {} \n".
                        format(k, epoch, name[0], positive_p, negtive_p, gt_label, output_label))

                # epoch cls metrics
            acc = true_num / total_num
            cls_precision = tp / (tp + fp + 1e-8)
            cls_recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + 1e-8)
            open(val_cls_log, 'a').write(
                "Fold: {} | Epoch: {}  | Val_Acc: {:4f} | Precision: {:4f} | Recall: {:4f} | F1_SCORE: {:4f} | \n".format(
                    k, epoch, acc, cls_precision, cls_recall, f1_score ))
        # epoch seg metrics
        writer.add_scalar('val_acc', acc, global_step=epoch)
        writer.add_scalar('val_cls_precision', cls_precision, global_step=epoch)
        writer.add_scalar('val_cls_recall', cls_recall, global_step=epoch)
        writer.add_scalar('val_cls_f1score', f1_score, global_step=epoch)
        writer.add_scalar('dice', dice, global_step=epoch)
        writer.add_scalar('jaccard', jaccard, global_step=epoch)
        writer.add_scalar('hd95', hd95, global_step=epoch)
        writer.add_scalar('val_seg_precision', seg_precision, global_step=epoch)
        writer.add_scalar('val_seg_sensitivity', seg_sensitivity, global_step=epoch)
        writer.add_scalar('val_seg_specificity', seg_specificity, global_step=epoch)
        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), check_point + '/Best_Dice.pth')
        if cls_recall > best_recall:
            best_recall = cls_recall
            torch.save(net.state_dict(), check_point + '/Best_Recall.pth')
        torch.save(net.state_dict(), check_point + '/model_fold{}_epoch{}.pth'.format(k, epoch))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
train_list = get_data_list("G:/tuFramework_data_store/Breast_s", ratio=0.8)
train_log = 'G:/tuFramework_data_store/cls_0523/train/trainLog.txt'
val_cls_log = 'G:/tuFramework_data_store/cls_0523/valid/valclsLog.txt'
val_path = 'G:/tuFramework_data_store/cls_0523/valid'
train_path = 'G:/tuFramework_data_store/cls_0523/train'
check_point = 'G:/tuFramework_data_store/cls_0523/checkpoints'
tensorboard_dir = 'G:/tuFramework_data_store/cls_0523/tensorboard_log'

if not os.path.exists(check_point):
    os.makedirs(check_point)
if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
information_line = '=' * 20 + ' MICCAI_DMF_Deep_Att_0228-TRAIN ' + '=' * 20 + '\n'
open(train_log, 'w').write(information_line)
information_line = '=' * 20 + ' MICCAI_DMF_Deep_Att_0228-VAL-SEG ' + '=' * 20 + '\n'
information_line = '=' * 20 + ' MICCAI_DMF_Deep_Att_0228-VAL-CLS ' + '=' * 20 + '\n'
open(val_cls_log, 'w').write(information_line)
K = 5
k_fold(K, train_list, train_log, val_cls_log,tensorboard_dir)

