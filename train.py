import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import torch

from network.RMNet import RMNet
from torch.optim.adamax import Adamax
from torchsummary import summary
from torch.backends import cudnn
import numpy as np
from time import *
from Utils import *
from Dataset import *
from Train_FCDC import *
from Loss import *
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

STEGO_DICT = {
    'DATASET':'/data/public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    tensorboard_save_dir = os.path.join(args.model_path, args.dataset)
    if os.path.exists(tensorboard_save_dir) == False:
        os.makedirs(tensorboard_save_dir)

    writer = SummaryWriter(log_dir=tensorboard_save_dir)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''model save dir and log'''
    args.log_path = os.path.join(args.model_path, args.log_name)
    args.write_log = write_log

    '''build dataset from args'''
    traing_set = STEGO_DICT['DATASET']
    train_dataloader, test_dataloader, val_dataloader = build_Dataloader(
        args, traing_set)
    
    '''optimizer and loss'''
    loss_cross = nn.CrossEntropyLoss().to('cuda')
    loss_affinity = Feature_Clustering_Loss(num_class=2, feat_dim= 256).to('cuda')
    loss_CC = Distance_Constraint_Loss().to('cuda')
    
    net = RMNet().to('cuda')
    params = list(net.parameters()) + [loss_affinity.centers]
    optimizer = Adamax(params, args.learning_rate)  #采用Adamax优化器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epochs)

    '''build model'''

    start_epoch = args.start_epoch
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1])
    if args.model_load_name != '':
        checkpoint = torch.load(os.path.join(args.model_load_name))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("加载模型成功！")
    else:
        net.apply(initWeights)
    #print(net)
    #print(summary(net.cuda(), (1, 1, 256, 256)))
    

    '''train'''
    best_test_acc = 0
    best_val_acc = 0
    args.write_log(
                args.log_path,
                "traing dataset is: " + traing_set
            )
    for epoch in range(start_epoch, args.max_epochs):
        args.write_log(
            args.log_path, 'epoch: {} lr: {}'.format(epoch, optimizer.state_dict()["param_groups"][0]["lr"])
        )

        net = train(
            epoch=epoch,
            net=net,
            optimizer=optimizer,
            loss_1 = loss_cross,
            loss_2 = loss_affinity,
            loss_3 = loss_CC,
            train_dataloader_cross = train_dataloader,
            args=args,
            writer = writer
        )

        val_acc = test(
            net=net,
            loss_func=loss_cross,
            test_dataloader=val_dataloader,
            args=args    
        )

        if epoch % 25 == 0:
            #save model checkpoint
            train_save_name = 'checkpoint_' + str(epoch)
            train_state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch}
            with open(os.path.join(args.model_path, train_save_name), 'wb') as out:  #写入二进制模式
                torch.save(train_state, out)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            with open(os.path.join(args.model_path, args.model_save_name), 'wb') as out:
                torch.save(state, out)

        args.write_log(
            args.log_path, 'epoch: {} best_val_acc: {}\n'.format(epoch, best_val_acc)
        )
        lr_scheduler.step()