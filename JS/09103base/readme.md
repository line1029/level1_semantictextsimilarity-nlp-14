# 0.9103 5epoch 체크포인트 로드
base plm model : klue/roberta-large  
Loss : L1  
lr : 1e-5  
epoch : 6  
batch_size : 16  
weight_decay : 0.01  

### 변경사항
없음  
lr : 1e-5로 초기화됨

### before
test_pearson(val set) : 0.9258
leaderboard score(test set, public) : 0.9103 
### after
test_pearson(val set) : 0.9281
leaderboard score(test set, public) : 0.9147 


### 이후 6epoch 더 돌렸을 땐 성능 하락