
### 13-001 

민수님 12-001코드 Data Augmentation 추가 시도 v2

ValueError: too many values to unpack (expected 2) 발생 -> 해결 못함


### 17-001
snunlp/koelectra-d, AdamW,	inversesqrtLR,	batch_size=32	epoch=10	lr=2e-05	warmup_ratio=0.30	decay=0.01	loss=L1

-> Val DT 기준 0.8320

-> batch_size 48 : OOM

### 17-002
snunlp/koelectra-d, AdamW,	inversesqrtLR,	batch_size=16	epoch=10	lr=2e-05	warmup_ratio=0.30	decay=0.01	loss=L1

-> Val DT 기준 0.8106

-> batch_size는 32가 효율적인 듯함.


### masked word prediction 기반 데이터 augmentation

Klue/Roberta Large Model 사용

오리지널 Val DT 기준 sentence_1, sentence_2 1100개 중 114개 문장(10.4%)에 대한 유의어를 교체

ex)	눈물겨운면도 있었지만 흥미진진했음. -> 지겨운면도 있었지만 흥미진진했음.

그러나 완전한 의미의 유의어 교체는 아니기 때문에, label값이 맞는지는 실험을 통해서 판단해야 함.

