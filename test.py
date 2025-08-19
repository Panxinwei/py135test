import joblib
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# 用户输入数据
def user_input_features():
    S1 = st.sidebar.slider('2_B', 0, 300, 102)
    S2 = st.sidebar.slider('3_C', 0.00, 10.00, 4.86)
    S3 = st.sidebar.slider('8_H', 0.00, 50.00, 4.07)
    S4 = st.sidebar.slider('19_S', 0.00, 10.00, 0.45)
    S5 = st.sidebar.slider('25_Y', 0.0, 10.0, 4.1)
    S6 = st.sidebar.slider('28_AB', 0, 80, 53)
    S7 = st.sidebar.slider('30_AD', 0.0, 90.0, 9.0)
    S8 = st.sidebar.slider('31_AE', 0.0, 1000.0, 285.7)
    data = {'2_B': 2_B, '3_C': 3_C,
            '8_H': 8_H, '19_S': 19_S,
            '25_Y': 25_Y, '28_AB': 28_AB,
            '30_AD': 30_AD, '31_AE': 31_AE
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

model=joblib.load('AdaBoost.pkl')


# 对输入数据进行分类，并进行展示
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.write(f"预测结果: {prediction[0]}")
