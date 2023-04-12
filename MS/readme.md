# KLUE STS Project
## 요약
### 모델 훈련
`train.py` 실행
### 예측
`inference.py` 실행

## 버전별 정리
### 04-10
- 베이스라인 코드 분석
- MSE loss 시도 (L1 -> MSE)
- Weight decay 파라미터 추가
- Warm up stage 추가 시도 -> 실패
### 04-11-001
- sweep 시도
  - 미션 2 코드와 sweep 공식 튜토리얼을 참고해 작성
  - KLUE 논문을 참고해서 lr, max_epoch, batch_size, weight_decay후보군 선택
  - warm up stage 다시 시도 -> 실패
### 04-11-002
- roberta-base 모델 시도
- batch_size = 32
- max_epoch = 15
- lr = 1e-5
- weight_decay = 0
- 16bit precision 시도
### 04-11-003
- roberta-base 모델의 sweep 시도
  - warm up stage 다시 시도 -> 성공, pytorch lightning의 scheduler 기본 인터벌을 epoch -> step으로 수정해서 해결
  - 여러가지 loss_func 시도(L1, MSE, Huber)
    - L1과 Huber에서 nan값이 나옴 -> 분모가 0 -> 모든 항이 0이거나 inf -> gradient가 vanishing했거나 exploding했나?
    - gradient clipping 추가, Huber의 경우 해결 된 것 같으나 L1의 경우 여전함
- seed_everything 함수로 시드 고정(잘 안되는것같음)
- warmup step을 계산하는 과정에서 total step을 계산할 필요가 있는데 len(dataset)이 실패, 아직 방법을 찾지 못함
### 04-12-001
- roberta-large 모델 시도
