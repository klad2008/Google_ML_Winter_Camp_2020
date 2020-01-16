import numpy as np
import torch
import argparse
import os
import torch.nn as nn

from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
import glas_dataset
import metrics, transformer
#from utils import logger
from datetime import datetime
from tqdm import tqdm
import cv2

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='efficientunet', help='model')
parser.add_argument('--batch-size', '-bs', default=8, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-2, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=15, type=int, help='epochs')
parser.add_argument('--save-dir', '-save-dir', default='./checkpoints', type=str, help='save-dir')
parser.add_argument('--interval', '-inv', default=1, type=int, help='val and save interval')

train_dir = '/home/charlesxujl/data/train/imgs/'
mask_dir = '/home/charlesxujl/data/train/labels/'

val_dir = '/home/charlesxujl/data/test/imgs/'
val_mask_dir = '/home/charlesxujl/data/test/labels/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    global args
    args = parser.parse_args()


def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr


def get_dynamic_lr(iter, maxiter):
    power = 0.9
    lr = args.lr * (1 - iter / maxiter) ** power
    return lr

'''
def get_lr(cur, epochs):
    if cur < int(epochs * 0.7):
        lr = args.lr
    else:
        lr = args.lr * 0.1
    return lr

'''
def plot(epoch, x_test, y_pred, target, name=None):
    batch_sz = x_test.size(0)
    x_test[:, 0].mul_(0.5).add_(0.5)
    x_test[:, 1].mul_(0.5).add_(0.5)
    x_test[:, 2].mul_(0.5).add_(0.5)

    x_test = x_test.data.cpu().numpy()
    #y_pred = y_pred.data.cpu().numpy()
    #target = target.data.cpu().numpy()

    for i in range(batch_sz):
        x = x_test[i]
        targ_mask = target[i]
        pred_mask = y_pred[i]
        
        orig = np.uint8(x * 255)
        gt = np.uint8(x * targ_mask * 255)
        new = np.uint8(x * pred_mask * 255)

        orig = np.transpose(orig, (1, 2, 0))
        gt = np.transpose(gt, (1, 2, 0))
        new = np.transpose(new, (1, 2, 0))

        # print(orig.shape, new.shape)

        cur_dir = os.path.join('./results/', str(epoch))
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

        if name is None:
            cv2.imwrite(os.path.join(cur_dir, str(i)+'_raw.png'),orig)
            cv2.imwrite(os.path.join(cur_dir, str(i)+'_gt.png'), gt)
            cv2.imwrite(os.path.join(cur_dir, str(i)+'_pred.png'),new)
        else:
            cv2.imwrite(os.path.join(cur_dir, str(name[i])+'_raw.png'),orig)
            cv2.imwrite(os.path.join(cur_dir, str(name[i])+'_gt.png'), gt)
            cv2.imwrite(os.path.join(cur_dir, str(name[i])+'_pred.png'),new)


def train():
    parse_args()
    print(args.model)

    # net
    net = import_module('models.%s' % args.model).get_model()
    print(net)
    #net = net.cuda()
    net = net.to(device)
    
    # data
    dataset = glas_dataset.dataset(train_dir, mask_dir,  mode='train')
    valset = glas_dataset.dataset(val_dir, val_mask_dir, mode='val')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    print('Train samples: ', args.batch_size * len(dataloader))
    #exit(0)

    # optim & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    criterion_seg = nn.CrossEntropyLoss()

    #save model and loss in a new dir
    save_name = args.model + '_' + str(args.lr) + '_' + str(args.batch_size)
    save_dir = os.path.join(args.save_dir, save_name + '_' + str(TIMESTAMP))
    os.mkdir(save_dir)
   
    # log = logger.Logger(save_dir)
    # log = logger.Logger('./logs')
    print('Creating directory :', '\t', save_dir)
	
    best_iou = .0

    #evaluate(0, net, valloader)
    #exit(0)

    for epoch in range(args.epochs):
        total_loss = .0
        train_acc = .0
        total_acc = .0
        total_iou = .0
        count = .0

        net.train()
        # lr = get_lr(epoch, args.epochs)
        lr = get_dynamic_lr(epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        pbar = tqdm(dataloader, ascii=True)
        for x_train, y_train in pbar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
    
            # y_pred, y_clf = net(x_train)
            y_pred = net(x_train)

            loss = criterion_seg(y_pred, y_train)

            prediction = torch.max(y_pred, 1)[1].data.cpu().numpy()
            target = y_train.data.cpu().numpy()
            result = metrics.accuracy_pixel_level(prediction, target)
            acc, iou = result[:2]
			
            cur_bs = x_train.size(0)
            total_loss += loss.item() * cur_bs
            total_acc += acc * cur_bs
            total_iou += iou * cur_bs
            # train_acc += torch.sum(y_grade == y_clf.max(1)[1]).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += cur_bs
            
            pbar.set_description('loss:%.4f'%(total_loss/count) + '|acc:%.4f'%(total_acc/count) + '|iou:%.4f'%(total_iou/count))

        total_loss /=  count
        total_acc /= count
        total_iou /= count

  
        print('Epoch = ', epoch, 'Loss = ', total_loss, ' Acc = ', total_acc, ' Iou=', total_iou)

        # log.scalar_summary('train_loss', total_loss, epoch)
        # log.scalar_summary('train_acc', train_acc, epoch)

        if (1 + epoch) % args.interval == 0:
            total_loss_val, val_acc, val_iou = evaluate(epoch, net, valloader)
            print('-' * 50)
            print('Val_loss = ', total_loss_val, ' Val Acc = ', val_acc, ' Val Iou=', val_iou)
            print('-' * 50)
            #log.scalar_summary('val_loss', total_loss_val, epoch)
            #log.scalar_summary('Dice A', score, epoch)
            #log.scalar_summary('val_acc', val_acc, epoch)
        if(1 + epoch) % args.interval == 0 and val_iou > best_iou:
            best_iou = val_iou
            print('Saving state')
            state = {'epoch': args.epochs,
                         'model_state': net.state_dict(),
                         'optimizer_state': optimizer.state_dict(), }
            torch.save(state, save_dir + save_name + str(epoch) + '.pkl')
            print('Done saving state')

def evaluate(epoch, net,  valloader):
    val_loss = .0
    val_acc = .0
    val_iou = .0
    count = .0
    net.eval()
    criterion_seg = nn.CrossEntropyLoss()
	
    with torch.no_grad():
        pbar = tqdm(valloader, ascii=True)
        for  x_train, y_train in pbar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
    
            # y_pred, y_clf = net(x_train)
            y_pred = net(x_train)

            loss = criterion_seg(y_pred, y_train)
            
            prediction = torch.max(y_pred, 1)[1].data.cpu().numpy()
            target = y_train.data.cpu().numpy()
            result = metrics.accuracy_pixel_level(prediction, target)
            acc, iou = result[:2]
            
            cur_bs = x_train.size(0)
                       
            if count == 0:
                plot(epoch, x_train, prediction, target) 
            
            val_loss += loss.item() * cur_bs
            val_acc += acc * cur_bs
            val_iou += iou * cur_bs
            count += x_train.size(0)
            
            pbar.set_description('loss:%.4f'%(val_loss/count) + '|acc:%.4f'%(val_acc/count) + '|iou:%.4f'%(val_iou/count))
        return val_loss/count, val_acc/count, val_iou/count


def inference():
    inf_dir = './results/src/'
    save_dir = './results/dst/'

    
    parse_args()
    # data
    valset = glas_dataset.dataset(val_dir, val_mask_dir, mode='inference')
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    # net = import_module('models.%s' % args.model).get_model()
    net = import_module('models.efficientunet').get_model()
    ckpt = torch.load('/home/charlesxujl/segmentation/checkpoints/SOTA/efficientunet_0.01_8_2020-01-15T05-14-04/efficientunet_0.01_82.pkl')
    # ckpt = torch.load('/home/charlesxujl/segmentation/checkpoints/SOTA/unet_0.01_8_2020-01-14T09-55-31/unet_0.01_813.pkl')

    net.load_state_dict(ckpt['model_state'])
    
    net = net.to(device)
    net.eval()

    for i, (x_train,name) in enumerate(valloader):
        print(i)
        x_train = x_train.to(device)
        pred = net(x_train)

        prediction = torch.max(pred, 1)[1].data.cpu().numpy()
        # for ease, just set ground-truth as prediction,
        plot(0, x_train, prediction, prediction, name) 
        
    print('Done inference')


if __name__ == '__main__':
    print(233)
    #train()
    inference()
