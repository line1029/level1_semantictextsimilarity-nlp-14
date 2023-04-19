import argparse

import pandas as pd
import numpy as np
import random

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
gc.collect()
torch.cuda.empty_cache()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


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
    folder_path = '/opt/ml/level1_semantictextsimilarity-nlp-14/SH'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--input_size', default=8, type=int)
    parser.add_argument('--hidden_size', default=16, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default=folder_path+'/data/dev_pred_input_size_8.csv')
    parser.add_argument('--dev_path', default=folder_path+'/data/dev_pred_input_size_8.csv')
    parser.add_argument('--test_path', default=folder_path+'/data/dev_pred_input_size_8.csv')
    parser.add_argument('--predict_path', default=folder_path+'/data/test_pred_input_size_8.csv')
    parser.add_argument('--loss_func', default="MSE")
    parser.add_argument('--project_name', default="STS_ensemble_blending_SH_is8")
    parser.add_argument('--run_name', default="r002")
    parser.add_argument('--seed', default=42, type=int) # seed를 argument로
    args = parser.parse_args()
    
    # seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    
    sweep_config = {
        'method': 'random',
        'parameters': {
            'batch_size':{
                'values': [16, 32, 64, 128, 275],
            },
            'learning_rate':{
                'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
            },
            'hidden_size':{
                'values': [16, 32, 64]
            },
            'loss_func':{
                'values': ['L1', 'MSE', 'Huber']
            }
        },
        'metric': {
            'name':'val_pearson',
            'goal':'maximize'
        }
    }
    
    def sweep_train(config=None):
        pl.seed_everything(args.seed)
        
        wandb.init(config=config)
        config = wandb.config

        model = BlendingEnsembleModel(
            args.train_path,
            args.predict_path,
            args.input_size,
            config.hidden_size,
            config.learning_rate,
            config.loss_func,
            config.batch_size
        )
        wandb_logger = WandbLogger(project=args.project_name + args.run_name,
                                   name=f'Blending_is8_lr{config.learning_rate}_hs{config.hidden_size}lf_{config.loss_func}_bs{config.batch_size}')

        trainer = pl.Trainer(accelerator='gpu', 
                             max_epochs=args.max_epoch, 
                             logger=wandb_logger,
                             val_check_interval=0.5,
                             log_every_n_steps=1,
                             callbacks=[
                                 LearningRateMonitor(logging_interval='step'),
                                 EarlyStopping(
                                     'val_loss',
                                     patience=5,
                                     mode='min',
                                     check_finite=True
                                     ),
                                 ModelCheckpoint(
                                     './save/',
                                     'blending_{val_pearson:.4f}',
                                     monitor='val_pearson',
                                     save_top_k=1,
                                     mode='max'
                                     )
                                 ]
                             )

        trainer.fit(model=model)

    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project_name + args.run_name,
        entity='boostcamp-sts-14'
    )
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=100
    )