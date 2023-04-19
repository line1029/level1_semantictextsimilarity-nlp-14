# 필요 라이브러리 설치
!pip install transformers
!pip install konlpy
!pip install torch
!pip install git+https://github.com/haven-jeon/PyKoSpacing.git
!pip install tqdm

# 호출
from transformers import AutoTokenizer, AutoModelForMaskedLM
from konlpy.tag import Kkma
import torch
import pandas as pd
from pykospacing import Spacing
from tqdm import tqdm

# Klue Roberta Large 모델과 토크나이저
model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SynonymAugment:
  
  def __init__(self,sentence):
    self.sentence = sentence
    self.VA_list = []
    self.masking = []
    self.tokens = []
    self.mask_ids = []
    self.augmented_sentence = []
    self.masked_sentences = []

  # pos태깅 후 형용사(VA)추출
  def VA_find(self):
    kkma = Kkma()
    pos = kkma.pos(self.sentence)
    for pos_tagged in pos:
      if pos_tagged[1] == "VA":
          self.VA_list.append(pos_tagged[0])

  def tokenmaker(self):
      self.tokens=tokenizer.tokenize(self.sentence)
  
  # VA 길이에 따른 토큰 추적
  ## '예쁜' 의 경우 VA->'예쁘' 따라서 '예'를 포함하고 있는 token 추적
  ## '많은' 의 경우 VA->'많' 따라서 '많'을 포함하고 있는 token 추적 
  def masking_set(self):
    for VA in self.VA_list:
      for token in self.tokens:
        if len(VA[0]) != 1:
            if VA[0][:-1] in str(token):
                self.masking.append(token)
        else:
            if VA[0] in str(token):
                self.masking.append(token)

  #단어의 위치를 찾는 경우에만 mask token 부여
    for word in self.masking:
      masks = self.sentence.find(word)
      if masks != -1:
          self.mask_ids.append(masks)


  def masking_sentence(self):
      for mask_idx in range(len(self.mask_ids)):
        self.masked_sentences.append(self.sentence[:self.mask_ids[mask_idx]]
                                + tokenizer.mask_token       
                                + self.sentence[self.mask_ids[mask_idx] + len(self.masking[mask_idx]):])

  def prediction(self):
  # 예측 모델
      for masked_sentence in range(len(self.masked_sentences)):
          input_ids = tokenizer.encode(self.masked_sentences[masked_sentence], return_tensors="pt")
          mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
          output = model(input_ids.to(device))[0]  
          softmax = torch.nn.Softmax(dim=-1)
          probs = softmax(output[0, mask_token_index, :]).detach().numpy()

  # 예측 결과 출력
          predicted_token = tokenizer.convert_ids_to_tokens(torch.argmax(output[0, mask_token_index, :], dim=1).tolist())[0]
          if predicted_token != "[UNK]" and predicted_token != "[PAD]":
              new_sentence = self.sentence[:self.mask_ids[masked_sentence]]                                           \
                              + predicted_token                                                                       \
                              + self.sentence[self.mask_ids[masked_sentence] + len(self.masking[masked_sentence]):]
              if new_sentence != self.sentence:
                  self.augmented_sentence.append(new_sentence)
        
      return print(self.augmented_sentence)

#csv읽기
df=pd.read_csv("dev.csv")
list1=df["sentence_1"].tolist()
list2=df["sentence_2"].tolist()

#augmentation 실행
aug_list1=[]
aug_list2=[]
for i in tqdm(range(len(list1))):
  augment1 = SynonymAugment(list1[i])
  augment1.tokenmaker()
  augment1.VA_find()
  augment1.masking_set()
  augment1.masking_sentence()
  augment1.prediction()

  augment2 = SynonymAugment(list2[i])
  augment2.tokenmaker()
  augment2.VA_find()
  augment2.masking_set()
  augment2.masking_sentence()
  augment2.prediction()

  aug_list1.append(augment1.augmented_sentence)
  aug_list2.append(augment2.augmented_sentence)

# 여러가지 후보 중 우선 첫번째 출력된 문장만 교체
## 모든 후보를 교체할 수 있게끔 수정 예정
for j in range(len(list1)):
  if aug_list1[j] != []:
    df["sentence_1"].replace(df["sentence_1"][j],aug_list1[j][0],inplace=True)
  if aug_list2[j] != []:
    df["sentence_2"].replace(df["sentence_2"][j],aug_list2[j][0],inplace=True)

df.to_csv('syn_aug_dev.csv', index=False)

#바뀐 문장 갯수
count=0
for i in aug_list1:
  if i != []:
    count += 1
for i in aug_list2:
  if i != []:
    count += 1

    
####TEST CODE####

# augment = SynonymAugment('특히 평소 뮤직채널에 멋진 영감을 불어넣어주시는!')
# augment.tokenmaker()
# augment.VA_find()
# augment.masking_set()
# augment.masking_sentence()
# augment.prediction()
# print(augment.VA_list)
# print(augment.masking)
# print(augment.tokens)
# print(augment.masked_sentences)
