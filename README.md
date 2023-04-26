# NAVER boostcamp NLP Lv. 1 STS 대회

**NLP-14조**

김민혁_T5031, 김세형_T5038, 이주형A_T5154, 이준선_T5157, 전민수_T5183

# 1. 프로젝트 개요

### **문장 간 유사도 측정**

의미 유사도 판별(Semantic Text Similarity, STS)이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 태스크

### 활용 장비 및 재료(개발 환경, 협업 툴 등)

- **서버**: AI Stage (NVIDIA V100 32GB)
- **IDE**: VSCode, Jupyter Lab

- **협업**: Git, GitHub, Slack
- **정리**: Notion, Google SpreadSheets

# 2. 팀 구성 및 역할

### 전민수_T5183

- PLM 모델 분석 및 모델별 성능 비교
- 데이터 분포에 따른 학습 데이터의 resampling 수행
- 선정된 PLM 모델로 STS 태스크 수행 모델 설계 및 튜닝, 그 중 성능이 높은 모델 기준으로 앙상블 모델 설계 및 튜닝

### 김민혁_T5031

- Data Augmentation - KLUE/Roberta large 기반 Synonym 교체
- Data Preprocessing - Konlpy 맞춤법 교정 시도

### 김세형_T5038

- Hyperparameter tuning tool 탐색 (Optuna, Sweep) 및 다수 모델 tuning
- K-fold cross validation 시도
- 다수의 학습 방법론 실험 및 검증

### 이주형A_T5154

- ******************************************시각화 및 전처리:****************************************** 데이터 분포 분석 및 시각화, 데이터 전처리
- **데이터 증강**: Pororo 기반 문장 패러프레이징

### 이준선_T5157

- hyperparameter tuning tool 적용 (sweep)
- learning rate scheduler 실험
- main code refactoring

# 3. 수행 절차 및 방법

![Untitled](https://user-images.githubusercontent.com/45084974/233902139-fb2e2053-cd6e-45ad-bf93-5a9282dd81f8.png)

# 4. 수행 결과

## 4.0. 프로젝트 구조도

```bash
.
|____code
| |____seed.py
| |____train.py
| |____ensemble_prepare.py
| |____ensemble_blending.py
| |____ckpt_to_pt.py
| |____sweep.py
| |____inference.py
|____README.md
```

## 4.1. EDA & Data Augmentation

### 데이터 분석 (EDA) 및 데이터 정제(Cleaning)

1. 데이터 소스(source) 별 라벨 분포(0~5) 시각화 
→ 데이터 Imbalance 확인, uniform distributional 하게 데이터 증강 및 샘플링 진행
2. 데이터 문장 내 UNK 토큰 통계적 분석 
→ 맞춤법 교정 + 이모티콘 제거 + 감정표현(ㅋ,ㅎ) 보정 + 문장부호 정규화 작업 진행

### **Masking 기반 유의어 교체 문장 생성**

1. 문장 별로 POS Tagging 후, VA(형용사)에 해당하는 토큰을 마스킹 (문장 본래의 의미 보존可)
2.  `klue/roberta-large` 모델을 활용해서 마스킹 토큰 예측
3. 예측한 토큰과 정답 토큰과 비교, 형태가 동일하지 않을 경우 예측한 토큰을 정답 토큰의 유의어로 간주

결과: Train 데이터 셋 3,108 sample 증가, Baseline 모델 기준 Pearson 점수 증가 (82.3 → 85.7)

### Pororo 기반 문장 패러프레이징

1. 카카오브레인 NLP 라이브러리 Pororo의 Seq2seq 기능 사용
2. 문장 내 토큰 개수 11개 이상인 경우에 Pororo 문장 패러프레이징 적용, 데이터 증강

결과: Train 데이터 셋 기존 대비 180% 증가, 모델 성능 향상은 확인하지 못함.

## 4.2. Base Code Analysis & Model Selection

### 베이스라인 코드 분석

- `pytorch lightning`, `transformers`라이브러리를 통해 모델 구현
- `transformers`에서 불러온 PLM모델을 이용해 fine tuning

### 모델 후보군

`klue/roberta`

- small, base, large 모델 시도

`xlm-roberta-large`

`monologg/koelectra-base-v3-discriminator`

`beomi/KcELECTRA-base`

`rurupang/roberta-base-finetuned-sts`

- paperswithcode에서 찾은 klue benchmark sota model

`snunlp/KR-ELECTRA-discriminator`

### 모델 선택

위 모델들을 하이퍼파라미터를 조정하며 여러번 학습, 최종 모델로 `klue/roberta-large`와`snunlp/KR-ELECTRA-discriminator` 모델 선정

### 시도했지만 적용 못한 모델

<aside>
💡 Base code의 `transformers.AutoModelForSequenceClassification`로  load되지 않았음.
다른 class로 load했지만 input error, train error등이 발생.
Huggingface 숙련도 부족으로 해결하지 못함.

</aside>

- `kykim/bertshared-kor-base`
- `gogamza/kobart-base-v2`

## 4.3. Hyperparameter Tuning

### Tuning Configuration

<aside>
💡 KLUE 논문을 참고하여 configuration을 선정하였음

</aside>

- **Learning Rate**: 5e-6, 6e-6, 8e-6, 1e-5, 2e-5, 3e-5, 5e-5
- **Loss Function**: MSE, L1, Huber
- **Max Epoch**: 4, 5, 6, 8
- **Batch Size**: 4, 8, 16, 20, 32, 48, 60, 64

- **Weight Decay**: 0, 0.01, 0.02, 0.05, 0.1
- **Warm Up Ratio**: 0, 0.1, 0.2, 0.3, 0.45, 0.6
- **LR Scheduler**: Step, InverseSqrt, Constant, Lambda, Linear
- **Optimizer**: Adam, AdamW

### Hyperparameter Tuning: WandB - Sweep

<aside>
💡 다른 hyperparameter tuning 툴인 Optuna와 비교 후 사용하려 했으나, PyTorch Lightning의 최신 버전과 Optuna 모듈 간 버전 충돌로 추정되는 오류로 인해 WandB - Sweep만 사용

</aside>

- 단일 모델
    - Model selection 결과로 도출된 klue/roberta-large와 snunlp/KR-ELECTRA-discriminator 모델의 hyperparameter tuning을 Sweep을 통해 수행함
- 앙상블 모델
    - 최고 성능으로 도출된 8개 / 12개의 단일 모델들의 결과를 blending을 통해 앙상블하기 위해 2-layer linear network를 제작하였고, 각 앙상블 모델의 hyperparameter tuning을 Sweep을 통해 수행함

## 4.4. 단일 모델 평가 및 개선

<aside>
💡 K-fold cross validation의 경우, 대회 초반 그 당시 성능이 가장 높았던 klue/roberta-large 모델에 10-fold로 적용을 시도해보았으나, stratified k-fold가 아닌 일반 k-fold 방법론을 적용하여 학습 효율이 감소하는 결과를 확인.

</aside>

### klue/roberta-small, base, large

- Base code 모델로 RoBERTa-small이 사용됨
- 모델 크기는 각각 68M, 111M, 336M
- 모델 크기로 인해 최대 batch size는 작아졌으나, 성능은 큰 모델일수록 좋아짐
- 최대 `val_pearson`: 0.9341 (large)

### rurupang/roberta-base-finetuned-sts

- 최대 `val_pearson`: 0.9253

### snunlp/KR-ELECTRA-discriminator

- 모델 크기 109M
- 단일 모델 학습시 KLUE/RoBERTa-large와 비슷하거나 높은 성능 도출
- 최대 `val_pearson`: 0.9340

### monologg/KoELECTRA-base-v3-discriminator

- 최대 `val_pearson`: 0.9201

### beomi/KcELECTRA-base

- lr, scheduler, warmup_steps를 조정했으나 수렴이 되지 않아 적용 불발

### 데이터 재분석 / 샘플링 개선

<aside>
💡 EDA를 다시 진행한 결과 데이터 불균형 문제 발견

</aside>

- Ver. 1: 학습 데이터를 샘플링(Label 0 → 언더샘플링, 4.4이상 → 오버샘플링)
- Ver. 2: Ver. 1 데이터에서 label이 0인 데이터를 추가로 언더샘플링
- Ver. 3: Ver. 2 데이터에서 샘플링 후 저장하는 방식 사용 시 버려지는 데이터가 존재함을 확인해 매번 코드 상에서 샘플링하는 방식으로 변경
    - 각 점수별로 600개씩 중복을 허용하여 샘플링 (0.5단위의 데이터는 수가 적어 60개씩 샘플링)

### 코드 개선

- 샘플링을 `Dataloader` 안에서 진행하여 속도 및 데이터 품질 개선
- `precision=“16-bit mixed”`을 이용해 속도 개선
- `val_check_interval`을 조정해 학습 추이를 더 자세히 관찰
- Callback에 `EarlyStopping`, `ModelCheckpoint`를 추가해 학습 간편화
    - `ModelCheckpoint`를 상속해 점수가 일정 이상일 경우에만 모델을 저장하도록 개선

## 4.5. Ensemble

### 사용된 단일 모델

- 훈련시킨 `klue/roberta-large`모델과 `snunlp/KR-ELECTRA-discriminator`모델 중 점수가 높았던 모델 8가지를 선택

### Voting

- **Hard Voting**(단순평균)
    - 오히려 최종 점수가 감소하였음
- **Soft Voting**
    - 각 점수를 정수 단위로 반올림 한 후 F1 점수를 측정
    - 각 테스트 샘플에서 각 모델이 예측한 점수에 해당하는 F1점수들을 softmax하여 샘플과 내적
    - Public 기준 0.9299

- **Blending**
    - 간단한 MLP 모델을 구성하여 학습
    - 크기 64의 Hidden Layer 2개
    - Batch size : 32
    - Learning Rate : 1e-4
    - Epoch : 20
    - Loss : MSE
    - Public 기준 **0.9326**

# 5. 자체 평가 의견

### 잘했던 점

- 초기에 목표했던 점수를 지속적으로 갱신해 나갔음
- HuggingFace로 시도할 수 있는 가능한 한 많은 모델을 시도하였음
- PyTorch Lightning, HuggingFace, Sweep 등 생소한 툴들에 대해 성실하게 이해하고 활용하려고 시도하였음

### 프로젝트를 통해 배운 점

- 데이터 자체에 대한 분석을 늦게 시작했으며, 깊이있게 시도하지 못했음
- 역할 분담 및 진행의 체계화가 부족했으며, 진행 상황 정리 및 실시간 공유 등에 부족함이 있었음
    - Notion, GitHub Projects, SpreadSheets 등 각종 협업 툴에 대해 익숙하지 못했음.
    - 깃허브 브랜치 관리를 통해 main file 관리의 필요성을 배웠음.
- “왜 안되는가?” 에 대한 분석이 부족했음
    - 가설 설정 및 그에 대한 실험 검증 결과를 연관짓는 과정이 빈약했음
    - e.g., Loss function과 같은 하이퍼파라미터 선택에 대해 가설-검증 과정 없이 실험적으로 시도했음.
- 모델의 구조를 이해하거나 응용해보지 못하고 그대로 사용함
