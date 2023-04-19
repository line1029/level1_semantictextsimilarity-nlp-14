import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import numpy as np
import random
import wandb
from itertools import chain
from seed import *
from train import *



if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=6, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='~/data/train_sentence_swap.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    parser.add_argument('--weight_decay', default=0.02)
    parser.add_argument('--warm_up_ratio', default=0.3)
    parser.add_argument('--loss_func', default="MSE")
    args = parser.parse_args()

    candidates = ['klue_rl_L1_0005_val_pearson=0.9341.pt', 'klue_rl_MSE_0001_val_pearson=0.9274.pt',
                  'klue_rl_L1_001_9288.pt', 'klue_rl_MSE_0007_val_pearson=0.9335.pt',
                  'snunlp_0004_val_pearson=0.9340.pt', 'snunlp_0015_val_pearson=0.9314.pt',
                  'snunlp_0011_val_pearson=0.9312.pt', 'snunlp_MSE_001_val_pearson=0.9316.pt']
    # candidates = [
    #     'klue_rl_L1_0004_val_pearson=0.9314.pt', 'klue_rl_MSE_0001_val_pearson=0.9311.pt',
    #     'snunlp_0006_val_pearson=0.9309.pt', 'snunlp_0016_val_pearson=0.9312.pt'
    # ]

    train_pred = []
    # dev_pred = []
    # test_pred = []
    for model_path in candidates:
        model = torch.load('./save/' + model_path)
        if 'klue' in model_path:
            model_name = 'klue/roberta-large'
            batch_size = 20
        elif 'snunlp' in model_path:
            model_name = 'snunlp/KR-ELECTRA-discriminator'
            batch_size = 48
        
        train_dataloader = Dataloader(model_name, batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.train_path)
        
        # dev_dataloader = Dataloader(model_name, batch_size, args.shuffle, args.train_path, args.dev_path,
        #                     args.test_path, args.test_path)
        
        # test_dataloader = Dataloader(model_name, batch_size, args.shuffle, args.train_path, args.dev_path,
        #                     args.test_path, args.predict_path)
        
        trainer = pl.Trainer(accelerator='gpu')

        train_pred.append(torch.cat(trainer.predict(model=model, datamodule=train_dataloader)))
        # dev_pred.append(torch.cat(trainer.predict(model=model, datamodule=dev_dataloader)))
        # test_pred.append(torch.cat(trainer.predict(model=model, datamodule=test_dataloader)))
    
    train_pred = torch.stack(train_pred).transpose(0, 1)
    # dev_pred = torch.stack(dev_pred).transpose(0, 1)
    # test_pred = torch.stack(test_pred).transpose(0, 1)
    train = pd.DataFrame(train_pred, columns=candidates)
    train['label'] = pd.read_csv(args.train_path)['label']
    train.to_csv('train_train_pred_base8.csv', index=False)
    # dev = pd.DataFrame(dev_pred, columns=candidates)
    # dev['label'] = pd.read_csv(args.dev_path)['label']
    # dev.to_csv('dev_train_pred_base8.csv', index=False)
    # test = pd.DataFrame(test_pred, columns=candidates)
    # test.to_csv('test_train_pred_base8.csv', index=False)
