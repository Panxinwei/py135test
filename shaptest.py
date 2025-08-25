
'''====================导入Python库===================='''
import joblib
import matplotlib
import pandas as pd               #python科学计算库
import numpy as np                #Python的一个开源数据分析处理库。
import matplotlib.pyplot as plt   #常用Python画图工具
#from xgboost import XGBRegressor  # 导入 XGBRegressor 模型
from sklearn.model_selection import train_test_split # 数据划分模块
from sklearn.preprocessing import StandardScaler   # 标准化模块
#from sklearn.metrics import mean_squared_error, r2_score   #误差函数MSE,误差函数R^2,
#from sklearn.model_selection import GridSearchCV     #超参数网格搜索
import openpyxl
from sklearn.tree import DecisionTreeRegressor
import shap  # 导入SHAP模型解释工具
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
def main():
#    matplotlib.use('TKAgg')
#    plt.style.use('ggplot')
    '''========================导入数据========================'''
    data = pd.read_excel('C:/Users/Administrator/Desktop/data.xlsx')  # 读取xlsx格式数据
    # date = pd.read_csv('D:/复现/trainset_loop6.csv')   #读取csv格式数据
#    print(data.isnull().sum())  # 检查数据中是否存在缺失值
#    print(data.shape)  # 检查维度
#    print(data.columns)  # 数据的标签
    #data = data.drop(["PN", "AN"], axis=1)  # axis = 1表示对列进行处理，0表示对行
    Y, X = data['label'], data.drop(['label'], axis=1)  # 对Y、X分别赋值
    columns = X.columns

    '''=========================标准化========================'''
    # 利用StandardScaler函数对X进行标准化处理
#    scaler = StandardScaler()
#    X = scaler.fit_transform(X)
    '''====================划分训练集与测试集==================='''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    model=joblib.load('AdaBoost.pkl')

    '''=====================SHAP解释模型======================'''
    explainer = shap.KernelExplainer(model.predict,X_test)  # 传入训练好的模型。
    shap_values = explainer.shap_values(X_test)  # 这里拿验证数据集进行呈现。输入X_train拿训练数据集进行呈现。
#    print(shap_values)
 #   explainer = shap.TreeExplainer(model)

        # 或者使用 shap.TreeExplainer(model) 来计算树模型的 SHAP 值
 #   shap_values = explainer(X_test).values[:,:,1]
    plt.figure(1)
    shap.summary_plot(shap_values=shap_values,  # 一个数组，其中包含了样本的 Shapley 值，大小为 [n_samples, n_features]
                  features=X_test,  # 一个数组，其中包含了样本的特征矩阵。这需要与 shap_values 中的样本数量一致。
                  feature_names=columns,  # 特征名称列表，长度应该和 features 的列数相同。
                  max_display=25,  # 可选参数，用于指定要显示的最多特征数量。默认情况下，将显示所有特征。
                  plot_type="bar",
                  show=False# 可选参数，用于指定图形类型。可以设置为 ‘dot’ 或 ‘bar’。
                  )


    plt.figure(2)
    shap.summary_plot(shap_values=shap_values,  # 一个数组，其中包含了样本的 Shapley 值，大小为 [n_samples, n_features]
                      features=X_test,  # 一个数组，其中包含了样本的特征矩阵。这需要与 shap_values 中的样本数量一致。
                      feature_names=columns,  # 特征名称列表，长度应该和 features 的列数相同。
                      max_display=25,  # 可选参数，用于指定要显示的最多特征数量。默认情况下，将显示所有特征。
                      plot_type="dot",
                      show=False# 可选参数，用于指定图形类型。可以设置为 ‘dot’ 或 ‘bar’。
                      )


# SHAP dependence plot (每个特征对因变量的贡献)
    X_test.reset_index(drop=True, inplace=True)
    for i in range(len(columns)):

        # max_feature_value_where_shap_negative=-1
        # min_feature_value_where_shap_postivate=-1
        shap.dependence_plot(columns[i], shap_values, X_test, interaction_index=None,show=False)
        feature_values = X_test.iloc[:, i].values
        feature_namesreshape=feature_values.reshape(-1, 1)
        dt = DecisionTreeRegressor(max_depth=1)
        dt.fit(feature_namesreshape, shap_values)

        # 3. 提取分裂阈值
        threshold_value = dt.tree_.threshold[0]


        # # 假设您知道原始形状
        # n_samples = X_test.shape[0]  # 样本数量
        # n_features = X_test.shape[1]  # 特征数量
        #
        # # 重新调整形状
        # shap_values_2d = shap_values.reshape(n_samples, n_features)
        # negative_shap_mask = shap_values_2d[:,i] < 0
        #
        # negative_feature_values = feature_values[negative_shap_mask]
        #
        # # 在这些 SHAP 值小于 0 的数据点中，找到最大的特征值
        # if len(negative_feature_values) > 0:
        #     max_feature_value_where_shap_negative = negative_feature_values.max()
        #
        #
        # postivate_shap_mask = shap_values_2d[:,i] > 0
        # postivate_feature_values = feature_values[postivate_shap_mask]
        #
        # # 在这些 SHAP 值小于 0 的数据点中，找到最大的特征值
        # if len(postivate_feature_values) > 0:
        #     min_feature_value_where_shap_postivate = postivate_feature_values.min()
        #
        #
        # if max_feature_value_where_shap_negative>0 and min_feature_value_where_shap_postivate>0:
        #     medX= round((max_feature_value_where_shap_negative+min_feature_value_where_shap_postivate)/2,2)
        # else:
        #     medX=0
        # 2. 获取当前坐标系
        ax = plt.gca()
        medX=round(threshold_value,2)
        # 3. 添加y=0的水平辅助线
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='y=0')
        ax.axvline(x=medX, color='r', linestyle='--', linewidth=1, label='Mean')
        ax.text(medX, 0.1 ,  f'x = {medX}', transform=ax.get_xaxis_transform(),
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.5),
                fontsize=10)

#    shap.dependence_plot(columns[len(columns)-1], shap_values, X_test, interaction_index=None)
 #   ax = plt.gca()
 #   ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='y=0')
 #   ax.axvline(x=medX, color='r', linestyle='--', linewidth=1, label='Mean')


    shap.initjs()  # 需要这段代码


    shap.plots.force(explainer.expected_value, shap_values[1], X_test.iloc[0, :],matplotlib = True)

# summary plot是针对全部样本预测的解释，有两种图，‘bar’一种是取每个特征的shap values的平均绝对值来获得标准条形图，这个其实就是全局重要度，
    # ‘dot’另一种是通过散点简单绘制每个样本的每个特征的shap values，通过颜色可以看到特征值大小与预测影响之间的关系，同时展示其特征值分布。

    # feature_importance = pd.DataFrame()  #创建一个 DataFrame 来存储 feature_importance。
    # feature_importance['feature'] = columns  # 给 'feature' 赋予特征名称。
    # feature_importance['importance'] = np.abs(shap_values).mean(0) # 对shap values按照特征维度聚合计算平均绝对值。
    # feature_importance.sort_values('importance', ascending = False)  # 对 'importance' values 不升序排序并放入feature_importance。





if __name__ == '__main__':
    main()
