### Setup

import re, os, argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict

from transformers import AutoTokenizer
from hanspell import spell_checker
from pykospacing import Spacing
from soynlp.normalizer import *

### Functions Setting

class DataCleaning():
    def __init__(self, train_path, dev_path, test_path, save_path, save_name):
        self.save_path = save_path
        self.save_name = save_name
        
        self.train_data = pd.read_csv(train_path)
        self.dev_data = pd.read_csv(dev_path)
        self.test_data = pd.read_csv(test_path)

        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')
        self.spacing = Spacing()
        
    #################################################################################### 함수 세팅
    def find_unk_check(self, sentence):
        _encode = self.tokenizer(sentence)
        unk_tokens = []
        
        # 1. sentence 내에 unk가 있는가?
        if _encode['input_ids'].count(self.tokenizer.unk_token_id):
            unk_tok = []
            
            _decode = self.tokenizer.convert_ids_to_tokens(_encode['input_ids'])
            unk_idx = [i for i, token in enumerate(_decode) if token == self.tokenizer.unk_token]
            
            for _idx in unk_idx:
                char_index = _encode.token_to_chars(_idx)
                original_token = sentence[char_index.start:char_index.end]  # char_index 는 CharSpan(start=15, end=19) 형태로 리턴되더랍니다... 신기!
                
                unk_tok.append(original_token)
            
            # 1-1. unk token 보관 -> 나중에 이게 있는지 없는지 체크 해야함.
            if unk_tok:
                unk_tokens.append(unk_tok)
                
            # 2. spelling correction. unk있는 문장은 spelling check
            result = spell_checker.check(sentence).as_dict()
            checked = result['checked']
            
            # 3. 변화가 있었는지 체크. 변화가 없는 단어만 따로 놓기.
            still_unk = []
            for ut in unk_tok:
                if checked.find(ut) != -1:
                    still_unk.append(ut)
            
            # 3-1. 변화가 없다면 pykospacing을 적용해보고, 다시 spelling check를 진행한다.
            if still_unk:
                space_checked = self.spacing(checked)
                # 앜ㅋㅋ 과 같은 감정표현 정제 기능 추가
                emo_checked = emoticon_normalize(space_checked, num_repeats=2) # ex. 안됔ㅋ큐ㅠ -> 안돼ㅋㅋ ㅜ
                result = spell_checker.check(emo_checked).as_dict()
                checked = result['checked']
                
            # 3-2. 변화가 있다면 변화된 문장으로 고쳐넣고. 그래도 변화 안 되는 문장은 내버려둔다.
            checked_sentence = checked

            # 4. unk가 인식되는지 확인.
            _encode = self.tokenizer(checked_sentence)
            space_unk = []

            if _encode['input_ids'].count(self.tokenizer.unk_token_id):
                _decode = self.tokenizer.convert_ids_to_tokens(_encode['input_ids'])
                unk_idx = [i for i, token in enumerate(_decode) if token == self.tokenizer.unk_token]
                
                for _idx in unk_idx:
                    char_index = _encode.token_to_chars(_idx)
                    original_token = checked_sentence[char_index.start:char_index.end]  # char_index 는 CharSpan(start=15, end=19) 형태로 리턴되더랍니다... 신기!
                    space_unk.append(original_token)

        else:
            checked_sentence = sentence
            unk_tokens, space_unk = None, None

        return checked_sentence, unk_tokens, space_unk

    #################################################################################### 함수 세팅
    # re 라이브러리 사용해서 제거 및 교체
    def cleaning(self, sentence):
        cleaned_sentence = sentence

        # punctuation cleaning
        # cleaned_sentence = re.sub(end_pattern, ".", cleaned_sentence)
        cleaned_sentence = re.sub(r"[?]+", "?", cleaned_sentence)
        cleaned_sentence = re.sub(r"[.,!;…‥~]+", ".", cleaned_sentence)
        
        # ㅋㅎㅉㅊㅠㅜ 때문에 맞춤법이 손상되는 경우 
        if re.search(r"[ㅋㅎㅉㅊㅠㅜ큐쿠]+", cleaned_sentence):
            cleaned_sentence = re.sub(r"[ㅋㅎ]+", "ㅋㅋ ", cleaned_sentence)  # ㅋ이 붙어있는 글자 받침에 있던없던 바로 지워버려서, 뒷 글자는 보통 없으므로 띄워준다.
            #cleaned_sentence = re.sub(r"[ㅉ]+", "", cleaned_sentence)      # ㅉ 은 UNK로 인식된다.
            cleaned_sentence = re.sub(r"[ㅊ]+", "ㅊ ", cleaned_sentence)
            cleaned_sentence = re.sub(r"[ㅠㅜ]+", "ㅜ ", cleaned_sentence)
            cleaned_sentence = re.sub(r"[큐쿠ㅉㅡ]+", "", cleaned_sentence)

        # UNK 통계에 따른 임의적인 sub 기능 추가
        cleaned_sentence = re.sub(r"[(네넵)|(네네넵)|(넵넵)|(넵)]", "네", cleaned_sentence)
        
        # 높임표현 낮추기 기능 추가
        cleaned_sentence = re.sub(r"[(뵌)|(봰)]", "본", cleaned_sentence)
        cleaned_sentence = re.sub(r"[(봴)|(뵐)]", "볼", cleaned_sentence)
        cleaned_sentence = re.sub(r"[(뵀)|(뵜)]", "봤", cleaned_sentence)

        # 기본형인 '봬' '뵈'는 어미가 달라져서 임의로 교체하기 힘들다.
        #cleaned_sentence = re.sub(r"[(봬요)|(뵐게요)]", "봐요", cleaned_sentence)
        
        cleaned_sentence = re.sub(r"[.]+", ".", cleaned_sentence)
        cleaned_sentence = re.sub(r"[:_/\｀☼^+<>@&$#*()]+", " ", cleaned_sentence)
        
        # emoji cleaning
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        cleaned_sentence = emoji_pattern.sub(r'', cleaned_sentence)

        # strip
        cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence)
        cleaned_sentence = cleaned_sentence.strip()

        # 문장 내에 스페이스가 없으면 띄어쓰기 해주기.
        if cleaned_sentence.find(" ") == -1:
            result = spell_checker.check(cleaned_sentence).as_dict()
            cleaned_sentence = result['checked']

        return cleaned_sentence

    #################################################################################### 함수 세팅
    def data_cleaning(self, _name):                    # cleaning + correction + save
        # setting   
        if 'train' in _name:                        df = self.train_data
        elif 'dev' in _name or 'valid' in _name:    df = self.dev_data
        elif 'test' in _name:                       df = self.test_data
        
        df_copied = df.copy(deep=True)

        unk_tokens, still_unk_tokens = [], []
        
        for i, row in tqdm(df_copied.iterrows()):
            org_sent_1, org_sent_2 = row['sentence_1'], row['sentence_2']
            
            # punctuation cleaning
            punc_cleaned_sent_1 = self.cleaning(org_sent_1)
            punc_cleaned_sent_2 = self.cleaning(org_sent_2)
            
            # Find UNK - spelling correction - Spacing - spelling correction again
            checked_sent_1, _unk_tokens_1, _still_unk_tokens_1 = self.find_unk_check(punc_cleaned_sent_1)    # _unk_tokens_1 = 원래 unk로 인식되는 단어, 
            checked_sent_2, _unk_tokens_2, _still_unk_tokens_2 = self.find_unk_check(punc_cleaned_sent_2)    # still... = 스펠링 교정, spacing 한 뒤 다시 교정 과정을 거친 뒤에도 unk로 인식되는 단어.
            
            if _unk_tokens_1 is not None:           unk_tokens.extend(_unk_tokens_1)
            if _unk_tokens_2 is not None:           unk_tokens.extend(_unk_tokens_2)
            if _still_unk_tokens_1 is not None:     still_unk_tokens.extend(_still_unk_tokens_1)
            if _still_unk_tokens_2 is not None:     still_unk_tokens.extend(_still_unk_tokens_2)
            
            # 원래 문장과 다르면 저장
            if org_sent_1 != checked_sent_1:
                df_copied.loc[i, 'sentence_1'] = checked_sent_1
            
            if org_sent_2 != checked_sent_2:
                df_copied.loc[i, 'sentence_2'] = checked_sent_2

        # 혹시나 생기는 NaN 제거
        df_copied = df_copied.replace('', np.nan, regex=True)
        result = df_copied.dropna(axis=0, subset=['sentence_1', 'sentence_2'], inplace=False)
        
        _save = os.path.join(self.save_path, _name+"_"+self.save_name)
        result.to_csv(_save, index=False)

        print(f"{_name} data has still unk token: \n{still_unk_tokens}\nNumber of them: {len(still_unk_tokens)}\n")
        return still_unk_tokens

#################################################################################### 실행 파트

if __name__ == "__main__":
    # Parser setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default='../data/train.csv', type=str)
    parser.add_argument("--dev_path", default='../data/dev.csv', type=str)
    parser.add_argument("--test_path", default='../data/test.csv', type=str)
    parser.add_argument("--save_path", default='../test', type=str)
    parser.add_argument("--save_name", default='clean_correct.csv', type=str)
    args = parser.parse_args()
    
    dc = DataCleaning(train_path=args.train_path,
                      dev_path=args.dev_path,
                      test_path=args.test_path,
                      save_path=args.save_path, 
                      save_name=args.save_name)

    still_train_unk = dc.data_cleaning("train")
    still_dev_unk = dc.data_cleaning("dev")
    still_test_unk = dc.data_cleaning("test")

