# 04-12 기준 SOTA 모델을 변형
V1
base plm model : klue/roberta-large  
Loss : L1  
lr : 1e-5  
epoch : 6  
batch_size : 16  
weight_decay : 0.01  
warm_up_ratio : 0.3  

scheduler를 get_cosine_with_hard_restarts_schedule_with_warmup로 변경

### before
test_pearson(val set) : 0.9266  
leaderboard score(test set, public) : 0.9075  
### after
test_pearson(val set) : 0.92151


# 04-12 기준 SOTA 모델을 변형
V2
base plm model : klue/roberta-large  
Loss : L1  
lr : 1e-5  
epoch : 6  
batch_size : 16  
weight_decay : 0.01  
warm_up_ratio : 0.3  

### 변경사항
scheduler를 get_cosine_with_hard_restarts_schedule_with_warmup로 변경
label 0값 약2100중 1000개만 랜덤사용

### before
test_pearson(val set) : 0.9266  
leaderboard score(test set, public) : 0.9075  
### after
test_pearson(val set) : 0.92085


# 04-12 기준 SOTA 모델을 변형
V3
base plm model : klue/roberta-large  
Loss : L1  
lr : 1e-5  
epoch : 6  
batch_size : 16  
weight_decay : 0.01  
warm_up_ratio : 0.3  

### 변경사항
label 0값 약2100중 1000개만 랜덤사용

### before
test_pearson(val set) : 0.9266  
leaderboard score(test set, public) : 0.9075  
### after
test_pearson(val set) : 0.92016


# 04-12 기준 SOTA 모델을 변형
V4
base plm model : klue/roberta-large  
Loss : L1  
lr : 1e-5  
epoch : 6  
batch_size : 16  
weight_decay : 0.01  
warm_up_ratio : -

### 변경사항
label 0값 약2100중 1000개만 랜덤사용
lr scheduler 제거

### before
test_pearson(val set) : 0.9266  
leaderboard score(test set, public) : 0.9075  
### after
test_pearson(val set) : 0.91362

# 04-12 기준 SOTA 모델을 변형 현재 sota
## 5번째 체크포인트 사용
V5
base plm model : klue/roberta-large  
Loss : L1  
lr : 1e-5  
epoch : 6  
batch_size : 16  
weight_decay : 0.01  

### 변경사항
data : train_resampled_swap.csv
lr scheduler : torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

### before
test_pearson(val set) : 0.9266  
leaderboard score(test set, public) : 0.9075  
### after
test_pearson(val set) : 0.9258
leaderboard score(test set, public) : 0.9103  