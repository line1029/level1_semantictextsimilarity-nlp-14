
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
