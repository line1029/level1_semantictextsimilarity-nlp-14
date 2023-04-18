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
from seed import *
from train import Dataset, Dataloader, Model, CustomModelCheckpoint, ResampledDataloader




if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="klue/roberta-large", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=6, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--train_path', default='~/data/train_sentence_swap.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    parser.add_argument('--weight_decay', default=0.01)
    parser.add_argument('--warm_up_ratio', default=0.3)
    parser.add_argument('--loss_func', default="MSE")
    parser.add_argument('--run_name', default="001")
    parser.add_argument('--project_name', default="STS_klue_rl_resampled_sweep_L1_MS")
    parser.add_argument('--eda', default=True)
    args = parser.parse_args()


    ### HP Tuning
    # Sweep을 통해 실행될 학습 코드를 작성합니다.
    sweep_config = {
        'method': 'random', # random: 임의의 값의 parameter 세트를 선택
        'parameters': {
            'learning_rate':{
                'values':[1e-5, 8e-6, 6e-6, 5e-6]
            },
            'max_epoch':{
                'values':[6]
            },
            'batch_size':{
                'values':[16]
            },
            'model_name':{
                'values':[
                    'klue/roberta-large',
                    # 'monologg/koelectra-base-v3-discriminator',
                    # 'beomi/KcELECTRA-base',
                    # 'rurupang/roberta-base-finetuned-sts',
                    # 'snunlp/KR-ELECTRA-discriminator'
                ]
            },
            # 'warm_up_ratio':{
            #     'values':[0.3, 0.45, 0.6]
            # },
            'weight_decay':{
                'values':[0, 0.01]
            },
            'loss_func':{
                'values':["L1"]
            }
        },
        'metric': {
            'name':'val_pearson',
            'goal':'maximize'
        }
    }

    def foo():
        for i in range(1, 1000):
            yield i
    ver = foo()

    def sweep_train(config=None):
        
        with wandb.init(config=config) as run:
            config = wandb.config
            # seed
            seed = get_seed()
            set_seed(*seed)
            run.name = f"{config.loss_func}_{config.learning_rate}_{config.batch_size}_{config.weight_decay}_{config.max_epoch}_steplr(0.96)_seed:{'_'.join(map(str,seed))}"
            
            wandb_logger = WandbLogger(
                project=args.project_name
            )
            dataloader = ResampledDataloader(config.model_name, config.batch_size, args.shuffle, args.train_path, args.dev_path,
                                    args.test_path, args.predict_path)
            # warmup_steps = int((15900 // config.batch_size + (15900 % config.batch_size != 0)) * config.warm_up_ratio)
            model = Model(
                config.model_name,
                config.learning_rate,
                config.weight_decay,
                # warmup_steps,
                # total_steps,
                config.loss_func
            )

            trainer = pl.Trainer(
                precision="16-mixed",
                accelerator='gpu',
                reload_dataloaders_every_n_epochs=1,
                max_epochs=config.max_epoch,
                logger=wandb_logger,
                log_every_n_steps=1,
                val_check_interval=0.25,
                check_val_every_n_epoch=1,
                callbacks=[
                    LearningRateMonitor(logging_interval='step'),
                    EarlyStopping(
                        'val_pearson',
                        patience=8,
                        mode='max',
                        check_finite=False
                    ),
                    CustomModelCheckpoint(
                        './save/',
                        f'klue_rl_{next(ver):0>4}_{{val_pearson:.4f}}',
                        monitor='val_pearson',
                        save_top_k=1,
                        mode='max'
                    )
                ]
            )
            trainer.fit(model=model, datamodule=dataloader)
            trainer.test(model=model, datamodule=dataloader)
    
    # Sweep 생성

    sweep_id = wandb.sweep(
        sweep=sweep_config,     # config 딕셔너리를 추가합니다.
        project=args.project_name  # project의 이름을 추가합니다.
    )
    wandb.agent(
        sweep_id=sweep_id,      # sweep의 정보를 입력하고
        function=sweep_train,   # train이라는 모델을 학습하는 코드를
        count=80                 # 총 3회 실행해봅니다.
    )

    ###

