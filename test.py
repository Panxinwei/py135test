import joblib
import numpy as np
import pandas as pd

import streamlit as st
from matplotlib import pyplot as plt
import shap

# import shap



# 用户输入数据
def user_input_features():
    S1 = st.sidebar.slider('2_B', 0, 300, 101)
    S2 = st.sidebar.slider('3_C', 0.00, 10.00, 4.86)
    S3 = st.sidebar.slider('8_H', 0.00, 50.00, 4.07)
    S4 = st.sidebar.slider('19_S', 0.00, 10.00, 0.45)
    S5 = st.sidebar.slider('25_Y', 0.0, 10.0, 4.1)
    S6 = st.sidebar.slider('28_AB', 0, 80, 66)
    S7 = st.sidebar.slider('30_AD', 0.0, 90.0, 9.0)
    S8 = st.sidebar.slider('31_AE', 0.0, 1000.0, 285.7)
    data = {'2_B': S1, '3_C': S2,
            '8_H': S3, '19_S': S4,
            '25_Y': S5, '28_AB': S6,
            '30_AD': S7, '31_AE': S8
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

model=joblib.load('AdaBoost.pkl')



if st.button("预测"):
# 对输入数据进行分类，并进行展示
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    st.write(f"预测结果: {prediction[0]}")
    st.write(f"预测阳性概率: {(prediction_proba[0][1])*100:.2f}%")




    #def inverse_sigmoid(logit):
    #    return 1 / (1 + np.exp(-logit))


    def convert_to_probability(logit_shap_values,total_logit_impact,total_prob_impact):
      #  probabilities = inverse_sigmoid(logit_shap_values)
      #  normalized_probabilities = probabilities / np.sum(probabilities)
        normalized_probabilities = ((logit_shap_values/ total_logit_impact[1]) * total_prob_impact[1])*100
        return normalized_probabilities







#计算 SHAP 值
    #explainer = shap.KernelExplainer(model.predict, df)
    #shap_values = explainer.shap_values(df)
    #explainer = shap.TreeExplainer(model,model_output="raw")
    explainer = shap.TreeExplainer(model)

    # 或者使用 shap.TreeExplainer(model) 来计算树模型的 SHAP 值
    shap_values = explainer(df)

    #
    shap.initjs()  # 需要这段代码
    #
    #
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], df,matplotlib = True)
    # #shap.plots.force(explainer.expected_value, shap_values[0, :], df.values,matplotlib = True)
    # # shap.summary_plot(shap_values=shap_values,  # 一个数组，其中包含了样本的 Shapley 值，大小为 [n_samples, n_features]
    # #               features=df,  # 一个数组，其中包含了样本的特征矩阵。这需要与 shap_values 中的样本数量一致。
    # #               feature_names=df.columns,  # 特征名称列表，长度应该和 features 的列数相同。
    # #               max_display=25,  # 可选参数，用于指定要显示的最多特征数量。默认情况下，将显示所有特征。
    # #               plot_type="bar",
    # #               show=False# 可选参数，用于指定图形类型。可以设置为 ‘dot’ 或 ‘bar’。
    # #               )
    # # 显示 SHAP 图
    # plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    # st.image("shap_force_plot.png")

    # 提取单个样本的 SHAP 值和期望值
    sample_shap_values = shap_values[0]  # 提取第一个样本的 SHAP 值
    base_value_logit = explainer.expected_value  # 获取对应输出的期望值


# 将基准值转换为概率
    base_value_prob = 1 / (1 + np.exp(-base_value_logit))
# 计算某个样本 i 的 Logit 值
    shap_values_logit = explainer.shap_values(df)
    sample_logit = base_value_logit + sample_shap_values.values[:, 1].sum()
    total_logit_impact = sample_logit - base_value_logit
    total_prob_impact = prediction_proba[0][1] - base_value_prob

# 转换为概率值
    probabilities = convert_to_probability(sample_shap_values.values[:, 1],total_logit_impact,total_prob_impact)
    # 创建 Explanation 对象
    explanation = shap.Explanation(
        #values=sample_shap_values[:, 1],  # 选择特定输出的 SHAP 值
        values=probabilities,
        base_values=base_value_prob[1]*100,
        data=df.iloc[0].values,
        feature_names=df.columns.tolist()
    )

    # 保存为 HTML 文件
    shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))

    # 在 Streamlit 中显示 HTML
    st.subheader("模型预测的力图")
    with open("shap_force_plot.html", encoding='utf-8') as f:
        st.components.v1.html(f.read(), height=600)




#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(pd.DataFrame(df, columns=df.keys()))
