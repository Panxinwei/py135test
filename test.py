import joblib
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# 用户输入数据
def user_input_features():
    S1 = st.sidebar.slider('S1', 0, 300, 102)
    S2 = st.sidebar.slider('S2', 0.00, 10.00, 4.86)
    S3 = st.sidebar.slider('S3', 0.00, 50.00, 4.07)
    S4 = st.sidebar.slider('S4', 0.00, 10.00, 0.45)
    S5 = st.sidebar.slider('S5', 0.0, 10.0, 4.1)
    S6 = st.sidebar.slider('S6', 0, 80, 53)
    S7 = st.sidebar.slider('S7', 0.0, 90.0, 9.0)
    S8 = st.sidebar.slider('S8', 0.0, 1000.0, 285.7)
    data = {'2_B': S1, '3_C': S2,
            '8_H': S3, '19_S': S4,
            '25_Y': S5, '28_AB': S6,
            '30_AD': S7, '31_AE': S8
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

model=joblib.load('AdaBoost.pkl')


# 对输入数据进行分类，并进行展示
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.write(f"预测结果: {prediction[0]}")