# -*- coding: utf-8 -*-

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import torch

from network.RMNet import RMNet
from torchsummary import summary
from torch.backends import cudnn
import numpy as np
from time import *
from Utils import *
from Dataset_test import *
from Train_test_FCDC import *

STEGO_DICT = {
    'DATASET':'path',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''model save dir and log'''
    test_path = 'ckpt'
    args.log_path = os.path.join(test_path, args.log_name)
    args.write_log = write_log

    '''build dataset from args'''

    test_dataloader = build_Dataloader(
        args, STEGO_DICT['DATASET']
    )

    '''build model'''
    net = RMNet().to('cuda')
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=[0, 1])
    checkpoint = torch.load(os.path.join(test_path + '/model_best'))
    net.load_state_dict(checkpoint['net'])

    if any(param is not None for param in net.parameters()):
        print("模型权重加载成功！")
    else:
        print("模型权重加载失败。")
    print("加载成功！")
    #print("Alpha value:", net.alpha.item())  # 使用 .item() 获取标量值
    #print(net)
    #print(summary(net.cuda(), (1, 256, 256)))

    '''optimizer and loss'''
    loss = nn.CrossEntropyLoss().to('cuda')

    test_acc = test(
        net=net,
        loss_func=loss,
        test_dataloader=test_dataloader,
        args=args,
    )

    args.write_log(
        args.log_path, 'test_acc: {} \n'.format(test_acc)
    )
    #lr_scheduler.step()
