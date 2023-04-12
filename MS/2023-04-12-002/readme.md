# 04-22 기준 SOTA 모델
base plm model : klue/roberta-large
Loss : L1
lr : 1e-5
epoch : 6
batch_size : 16
weight_decay : 0.01
warm_up_ratio : 0.3

test_pearson(val set) : 0.9266
leaderboard score(test set, public) : 0.9075