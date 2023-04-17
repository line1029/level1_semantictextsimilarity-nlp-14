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


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=6, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='~/data/train.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    parser.add_argument('--weight_decay', default=0.02)
    parser.add_argument('--warm_up_ratio', default=0.3)
    parser.add_argument('--loss_func', default="MSE")
    args = parser.parse_args()
    candidates_paths = ['klue_rl_0001_val_pearson=0.9274.ckpt',  'klue_rl_0005_val_pearson=0.9341.ckpt',  'klue_rl_0007_val_pearson=0.9335.ckpt']
    for path in candidates_paths:
        if 'klue' in path:
            model_name = 'klue/roberta-large'
        elif 'rurupang' in path:
            model_name = 'rurupang/roberta-base-finetuned-sts'
        elif 'snunlp' in path:
            model_name = 'snunlp/KR-ELECTRA-discriminator'
        dataloader = Dataloader(model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
        model = Model.load_from_checkpoint("./save/" + path)
        torch.save(model, "./save/" + path[:-4] + "pt")
