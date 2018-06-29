# /usr/bin/env python
# -*- coding: UTF-8 -*-

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf

from trainer import *
from trainer256 import *
from config import get_config
from utils import prepare_dirs_and_logger, save_config

import pdb, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(config):
    """
    config参数：Namespace(D_arch='DCGAN', batch_size=1, beta1=0.5, beta2=0.999, ckpt_path=None, conv_hidden_num=128, 
    d_lr=2e-05, data_dir='data', data_format='NCHW', dataset='DF_train_data', g_lr=2e-05, gamma=0.5, gpu=0, 
    grayscale=False, img_H=256, img_W=256, is_train=True, lambda_k=0.001, load_path='', log_dir='logs', 
    log_level='INFO', log_step=200, lr_update_step=50000, max_step=80, model=11, model_dir='path_to_directory_of_model',
     num_log_samples=3, num_worker=4, optimizer='adam', pretrained_path=None, random_seed=123, sample_per_image=64, 
     save_model_secs=1000, split='train', start_step=0, test_data_path=None, test_one_by_one=False, use_gpu=True, 
     z_num=2)

    """
    prepare_dirs_and_logger(config)

    if config.gpu > -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    config.data_format = 'NHWC'
    trainer = None
    if 1 == config.model:
        print("使用PG2()，即Market-1501 数据库，并初始化")
        trainer = PG2(config)
        trainer.init_net()
    elif 11 == config.model:
        print("使用PG2_256()，即DeepFashion数据库，并初始化")
        trainer = PG2_256(config)
        trainer.init_net()
        
    if config.is_train:
        print("开始训练")
        save_config(config)  # 存储参数到json文件
        trainer.train()  # 开始训练
    else:
        print("开始测试")
        if not config.load_path:
            raise Exception("[!] 没有指定 `load_path` 用于读取预训练的模型")
        trainer.test()


if __name__ == "__main__":
    print("主进程开启")
    config, unparsed = get_config()
    main(config)
