from koeda import EDA, AEDA
import pandas as pd
import numpy as np
import random
import os


eda = EDA("Mecab")
aeda = AEDA("Mecab")


if __name__ == "__main__":
    seed = 13256
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv("~/data/train.csv")
    l = [df.copy() for _ in range(6)]
    for idx, data in enumerate(l):
        if idx&1:
            data["sentence_1"] = aeda(eda(list(data["sentence_1"])))
        else:
            data["sentence_2"] = aeda(eda(list(data["sentence_2"])))
    
    df_augmented = pd.concat([df, *l])
    df_augmented.to_csv("~/data/train_augmented.csv", index=False)
