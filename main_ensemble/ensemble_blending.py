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


class BlendingEnsembleModel(pl.LightningModule):
    def __init__(self, train_path, test_path, input_size, hidden_size, lr, loss_func, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        if loss_func == "MSE":
            self.loss_func = torch.nn.MSELoss()
        elif loss_func == "L1":
            self.loss_func = torch.nn.L1Loss()
        elif loss_func == "Huber":
            self.loss_func = torch.nn.HuberLoss()
        self.train_path = train_path
        self.test_path = test_path
        self.shuffle = True
        self.batch_size = batch_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
        return optimizer
    
    def preprocessing(self, data):
        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data['label'].values.tolist()
            inputs = data.drop('label', axis=1).values.tolist()
        except:
            targets = []
            inputs = data.values.tolist()

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            train = pd.read_csv(self.train_path)

            train_inputs, train_targets = self.preprocessing(train)
            self.train_dataset = self.val_dataset = Dataset(train_inputs, train_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.train_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.test_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)



if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--input_size', default=8, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--train_path', default='~/data/dev_pred_input_size_8.csv')
    parser.add_argument('--dev_path', default='~/data/dev_pred_input_size_8.csv')
    parser.add_argument('--test_path', default='~/data/dev_pred_input_size_8.csv')
    parser.add_argument('--predict_path', default='~/data/test_pred_input_size_8.csv')
    parser.add_argument('--loss_func', default="MSE")
    args = parser.parse_args()

    
    # seed
    seed = get_seed()
    set_seed(*seed)

    # dataloader와 model을 생성합니다.
    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    wandb_logger = WandbLogger(
        project='STS-Ensemble2',
        name=f'Blending_{args.input_size}_lr:{args.learning_rate}_hs:{args.hidden_size}_bs:{args.batch_size}_epoch_{args.max_epoch}_seed:{"_".join(map(str, seed))}',
        entity='boostcamp-sts-14'
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.max_epoch,
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=0.25,
        check_val_every_n_epoch=1,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # EarlyStopping(
            #     'val_pearson',
            #     patience=8,
            #     mode='max',
            #     check_finite=False
            # ),
            CustomModelCheckpoint(
                './save/',
                'blending_{val_pearson:.4f}',
                monitor='val_pearson',
                save_top_k=1,
                mode='max'
            )
        ]
    )

    # Blending part
    # 저장된 모델로 블렌딩 앙상블, 예측을 진행합니다.
    model = BlendingEnsembleModel(
        args.train_path,
        args.predict_path,
        args.input_size,
        args.hidden_size,
        args.learning_rate,
        args.loss_func,
        args.batch_size
    )


    trainer.fit(model=model)


    predictions = trainer.predict(model=model)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('~/data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(f'Blending_{args.input_size}_lr_{args.learning_rate}_hs_{args.hidden_size}_bs_{args.batch_size}_epoch_{args.max_epoch}_seed_{"_".join(map(str, seed))}.csv', index=False)
