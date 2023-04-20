import argparse
import torch

from seed import *
from train import *


if __name__ == "__main__":
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='~/data/train.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    args = parser.parse_args()

    # 변환할 모델 리스트
    candidates_paths = [
        'snunlp_0016_val_pearson=0.9312.ckpt'
    ]

    for path in candidates_paths:
        if 'klue' in path:
            model_name = 'klue/roberta-large'
            batch_size = 20
        elif 'snunlp' in path:
            model_name = 'snunlp/KR-ELECTRA-discriminator'
            batch_size = 48

        dataloader = Dataloader(model_name, batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        # ckpt -> pt
        model = Model.load_from_checkpoint("./save/" + path)
        torch.save(model, "./save/" + path[:-4] + "pt")
