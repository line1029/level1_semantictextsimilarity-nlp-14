import streamlit as st
import torch
import pytorch_lightning as pl
from train import Dataloader, Dataset
import pandas as pd

sentence1 = st.text_input("첫 번째 문장")

sentence2 = st.text_input("두 번째 문장")


class StreamlitDataloader(Dataloader):
    def setup(self, stage='fit'):
        if stage != 'fit':
            # 평가데이터 준비
            predict_data = pd.DataFrame({'id':"", 'sentence_1':[sentence1], 'sentence_2':[sentence2], })
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])


@st.cache_resource
def load_model():
    model = torch.load("snunlp_MSE_001_val_pearson=0.9316.pt")
    trainer = pl.Trainer(accelerator='gpu')
    return model, trainer

if st.button("유사도 측정"):
    model, trainer = load_model()

    prediction = trainer.predict(model=model, datamodule=StreamlitDataloader("snunlp/KR-ELECTRA-discriminator", 1, True, "", "", "", ""))
    st.write(prediction)