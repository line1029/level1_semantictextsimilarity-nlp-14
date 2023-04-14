import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
import wandb
import numpy as np
import random
from train import Dataset, Dataloader, Model




if __name__ == '__main__':
    # seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="snunlp/KR-ELECTRA-discriminator", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=4, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--train_path', default='~/data/train_resampled_swap.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    parser.add_argument('--weight_decay', default=0.01)
    parser.add_argument('--warm_up_ratio', default=0.3)
    parser.add_argument('--loss_func', default="MSE")
    parser.add_argument('--run_name', default="001")
    parser.add_argument('--project_name', default="STS_resample_swap")
    parser.add_argument('--eda', default=True)
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        # precision="16-mixed",
        accelerator='auto',
    )

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    candidates = ['snunlp_MSE_001_9304.pt', 'snunlp_MSE_002_9285.pt', 'klue_ra_L1_001_9288.pt', 'klue_ra_L1_002_9282.pt']
    predictions = torch.zeros(1100)
    for model_name in candidates:
        model = torch.load(model_name)
        predictions += torch.cat(trainer.predict(model=model, datamodule=dataloader)) / len(candidates)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = [float(i) for i in predictions]
    for idx, val in enumerate(predictions):
        k = int(round(val*10))
        if k % 2 == 0 or k % 5 == 0:
            predictions[idx] = round(val, 1)
            continue
        if int(val*10) % 2 == 0:
            predictions[idx] = (k - 1) / 10
        else:
            predictions[idx] = (k + 1) / 10

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('~/data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('output.csv', index=False)
