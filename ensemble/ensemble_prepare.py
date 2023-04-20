import argparse
import pytorch_lightning as pl
import pandas as pd
import torch

from seed import *
from train import *


# output 만들 dataset 경로 설정
def set_path_to_predict(args):
    predict_path = None
    if args.predict == "train":
        predict_path = args.train_path
    elif args.predict == "dev":
        predict_path = args.test_path
    else:  # args.predict == "test":
        predict_path = args.predict_path

    return predict_path


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', default=True)
    parser.add_argument(
        '--train_path', default='~/data/train_sentence_swap.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    parser.add_argument('--predict', default="test",
                        type=str)  # output 만들 dataset (train, dev, test)
    args = parser.parse_args()

    # 불러올 모델들의 pt파일
    candidates = ['klue_rl_L1_0005_val_pearson=0.9341.pt', 'klue_rl_MSE_0001_val_pearson=0.9274.pt',
                  'klue_rl_L1_001_9288.pt', 'klue_rl_MSE_0007_val_pearson=0.9335.pt',
                  'snunlp_0004_val_pearson=0.9340.pt', 'snunlp_0015_val_pearson=0.9314.pt',
                  'snunlp_0011_val_pearson=0.9312.pt', 'snunlp_MSE_001_val_pearson=0.9316.pt']
    # candidates = [
    #     'klue_rl_L1_0004_val_pearson=0.9314.pt', 'klue_rl_MSE_0001_val_pearson=0.9311.pt',
    #     'snunlp_0006_val_pearson=0.9309.pt', 'snunlp_0016_val_pearson=0.9312.pt'
    # ]

    prediction = []

    for model_path in candidates:
        model = torch.load('./save/' + model_path)
        if 'klue' in model_path:        # klue/roberta-large 모델로 predict
            model_name = 'klue/roberta-large'
            batch_size = 20
        elif 'snunlp' in model_path:    # snunlp/KR-ELECTRA-discriminator 모델로 predict
            model_name = 'snunlp/KR-ELECTRA-discriminator'
            batch_size = 48

        # output 만들 dataset load
        dataloader = Dataloader(model_name, batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, set_path_to_predict(args))
        # trainer 설정
        trainer = pl.Trainer(accelerator='gpu')
        # predict한 뒤 prediction에 append
        prediction.append(torch.cat(trainer.predict(
            model=model, datamodule=dataloader)))
    # transpose
    prediction = torch.stack(prediction).transpose(0, 1)
    # output 만들기
    output = pd.DataFrame(prediction, columns=candidates)
    if args.predict != "test":
        output['label'] = pd.read_csv(set_path_to_predict(args))['label']
    output.to_csv(
        f'{args.predict}_train_pred_base{len(candidates)}.csv', index=False)
