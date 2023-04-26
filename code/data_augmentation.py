### Setup
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter, OrderedDict

from transformers import AutoTokenizer
from pororo import Pororo


class DataAugmentation():
    def __init__(self, ):
        

    # functions
    def paraphrased(text):
        tok = tokenizer(text)['input_ids']
        if len(tok) >= 11:
            paraphrased = paraphrasing(text,
                            beam=5,
                            len_penalty=0.7,
                            no_repeat_ngram_size=4,
                            top_k=50,
                            top_p=0.7
                            )
            return paraphrased
        else:
            return

    def continue_condition(score, zero_count):
        flag = True
        
        if score < 1.0:
            zero_count = zero_count + 1
            if zero_count % 33 == 0:            # 3% 만 증강
                flag = False
        else:
            flag = False
        # elif score >= 1.0 and score < 2.0:
        #     one_count = one_count + 1
        #     if one_count % 3 != 0:              # 66% 만 증강
        #         flag = True
        
        return flag, zero_count

    def aug_paraphrase(df):
        df_copied = df.copy(deep=True)
        aug_1, aug_2, label, blabel = [], [], [], []
        zero_count = 0
        
        for i, item in tqdm(df_copied.iterrows()):
            # 1. 행 하나 읽고
            org_1, org_2 = item['sentence_1'], item['sentence_2']
            org_label, org_blabel = item['label'], item['binary-label']
            
            # 1-1. 0~1은 좀 많이 안 증가시키고 싶다. 0 이면 80
            flag, zero_count = continue_condition(org_label, zero_count)
            if flag:
                continue
            
            # 2. org_1에 대해
            par_1, par_2 = paraphrased(org_1), paraphrased(org_2)
            
            if par_1:
                
                # 2-1-1. (ps1, org s2) + org label
                aug_1.append(par_1)
                aug_2.append(org_2)
                label.append(org_label)
                blabel.append(org_blabel)
                
                # 전체 약 9300 개에서 토큰개수 11개이상(평균이 13개정도)이 12007 개. 여기서 5 라벨은 114 개. 2배 증가 + 3000개 증가
                # 1-3-2. org1 - ps1 : 5.0으로 설정
                if i%7 == 0 and org_label != 5.0:
                    aug_1.append(org_1)
                    aug_2.append(par_1)
                    label.append(5.0)
                    blabel.append(1.0)
            
            # 3. org_2에 대해
            if par_2:
                
                # 3-1-1. (org s1, ps2) + org label
                aug_1.append(org_1)
                aug_2.append(par_2)
                label.append(org_label)
                blabel.append(org_blabel)
                
                # 3-1-2. org2 - ps2 : 5.0으로 설정.
                if i%13 == 0 and org_label != 5.0:
                    aug_1.append(par_2)
                    aug_2.append(org_2)
                    label.append(5.0)
                    blabel.append(1.0)
            # if i%10 == 0:
            #     print(f"aug1: {aug_1}\naug2: {aug_2}")
            
        # 4. DataFrame
        _id = ["paraphrased"] * len(aug_1)
        aug_df = pd.DataFrame({'id': _id,
                            'sentence_1': aug_1,
                            'sentence_2': aug_2,
                            'label': label,
                            'binary-label': blabel,
                            })
        concated_df = pd.concat([df_copied, aug_df])

        # 5. save
        concated_df.to_csv("./train_paraphrased.csv", index=False)
        aug_df.to_csv("./train_parap_onlyaug.csv", index=False)
        
        return concated_df, aug_df

    # swapping 하는 함수
    def data_swap(df):
        # org df
        df_copied = df.copy(deep=True)
        
        # swap df
        df_swap = df.copy(deep=True)

        df_swap = df_swap[['id', 'source', 'sentence_2', 'sentence_1', 'label', 'binary-label']]
        df_swap = df_swap.rename(columns={'sentence_1': 'sentence_2', 'sentence_2': 'sentence_1'})

        concated_df = pd.concat([df_copied, df_swap])
        concated_df.to_csv("./train_paraphrased_swap.csv")
        return concated_df


if __name__ == "__main__":
    train_path = './data/train_clean_correct.csv'
    dev_path = './data/dev_clean_correct.csv'

    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')

    ### Functions
    paraphrasing = Pororo(task="pg", lang="ko")

    train_paraphrased, train_parap_onlyaug = aug_paraphrase(train_data)
    train_paraphrased_swap = data_swap(train_paraphrased)