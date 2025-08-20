import joblib
import streamlit as st
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
#import shap
#shap.initjs()
# 用户输入数据
def user_input_features():
    S1 = st.sidebar.slider('2_B', 0, 300, 101)
    S2 = st.sidebar.slider('3_C', 0.00, 10.00, 4.86)
    S3 = st.sidebar.slider('8_H', 0.00, 50.00, 4.07)
    S4 = st.sidebar.slider('19_S', 0.00, 10.00, 0.45)
    S5 = st.sidebar.slider('25_Y', 0.0, 10.0, 4.1)
    S6 = st.sidebar.slider('28_AB', 0, 80, 53)
    S7 = st.sidebar.slider('30_A', 0.0, 90.0, 9.0)
    S8 = st.sidebar.slider('31_AE', 0.0, 1000.0, 285.7)
    data = {'2_B': S1, '3_C': S2,
            '8_H': S3, '19_S': S4,
            '25_Y': S5, '28_AB': S6,
            '30_A': S7, '31_AE': S8
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

model=joblib.load('AdaBoost.pkl')




# 对输入数据进行分类，并进行展示
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.write(f"预测结果: {prediction[0]}")
st.write(f"预测阳性概率: {(prediction_proba[0][1])*100:.2f}%")



#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(pd.DataFrame(df, columns=df.keys()))
