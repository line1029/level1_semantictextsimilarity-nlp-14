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
    
    
class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=128)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self, num_workers=8):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle, num_workers=num_workers)

    def val_dataloader(self, num_workers=8):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=num_workers)

    def test_dataloader(self, num_workers=8):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=num_workers)

    def predict_dataloader(self, num_workers=8):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=num_workers)
    
    
class Model(pl.LightningModule):
    def __init__(self, model_name, lr, loss_func, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 손실함수를 호출합니다.
        if loss_func == "MSE":
            self.loss_func = torch.nn.MSELoss()
        elif loss_func == "L1":
            self.loss_func = torch.nn.L1Loss()
        elif loss_func == "Huber":
            self.loss_func = torch.nn.HuberLoss()
            
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
        return [optimizer], [scheduler]
    
    
if __name__ == '__main__':
    folder_path = '/opt/ml/level1_semantictextsimilarity-nlp-14/SH'
    # seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-base', type=str)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default=folder_path+'/data/train_resampled_swap.csv')
    parser.add_argument('--dev_path', default=folder_path+'/data/dev.csv')
    parser.add_argument('--test_path', default=folder_path+'/data/dev.csv')
    parser.add_argument('--predict_path', default=folder_path+'/data/test.csv')
    parser.add_argument('--weight_decay', default=0.1)
    parser.add_argument('--loss_func', default="MSE")
    parser.add_argument('--run_name', default="_r001")
    parser.add_argument('--project_name', default="STS_roberta_base_001")
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, 
                            args.batch_size, 
                            args.shuffle, 
                            args.train_path, 
                            args.dev_path, 
                            args.test_path, 
                            args.predict_path
                           )
    
    model = Model(
            args.model_name,
            args.learning_rate,
            args.loss_func,
            args.weight_decay
            )

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accelerator = 'gpu' if device == torch.device('cuda') else 'cpu'
    trainer = pl.Trainer(precision="16-mixed", 
                             accelerator=accelerator, 
                             max_epochs=args.max_epoch, 
                             val_check_interval=0.5,
                             log_every_n_steps=1,
                             callbacks=[
                                 LearningRateMonitor(logging_interval='step'),
                                 EarlyStopping(
                                     'val_loss',
                                     patience=20,
                                     mode='min',
                                     check_finite=True
                                     ),
                                 ModelCheckpoint(
                                     './save/',
                                     '{epoch}-{step}-{val_pearson:.4f}',
                                     monitor='val_pearson',
                                     save_top_k=1,
                                     mode='max'
                                     ),
                                 ]
                             )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    model_name = folder_path + "/models/model_temp.pt"
    torch.save(model, model_name)