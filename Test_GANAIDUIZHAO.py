import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from io import BytesIO
import base64
import shap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from datetime import datetime
import warnings
import os
import sys


# 设置中文字体，解决中文显示问题
def setup_chinese_font():
    """设置中文字体以解决显示乱码问题"""
    # 检查操作系统类型
    if sys.platform.startswith('win'):
        # Windows系统
        font_paths = [
           # 'C:/Windows/Fonts/simhei.ttf',  # 黑体
           # 'C:/Windows/Fonts/simkai.ttf',  # 楷体
           # 'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'msyh.ttc',  # 微软雅黑
        ]
    elif sys.platform.startswith('linux'):
        # Linux系统
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
        ]
    else:
        # macOS系统
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # 苹方
            '/System/Library/Fonts/STHeiti Light.ttc',  # 华文黑体
        ]

    # 尝试添加中文字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                font_name = fm.FontProperties(fname=font_path).get_name()
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                matplotlib.rcParams['axes.unicode_minus'] = False
                return True
            except:
                continue

    # 如果找不到中文字体，使用默认设置
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False


# 初始化中文字体
setup_chinese_font()

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="医疗预测分析系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
    }
    .input-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<h1 class="main-header">🏥 医疗预测分析系统</h1>', unsafe_allow_html=True)

# 初始化session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame()
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# 侧边栏设置
with st.sidebar:
    st.markdown("## ⚙️ 系统设置")

    # 模型上传
    st.markdown("### 模型管理")
    use_default_model = st.checkbox("使用默认模型", value=True)

    if not use_default_model:
        uploaded_model = st.file_uploader("上传模型文件", type=['pkl', 'joblib', 'sav'])
        if uploaded_model:
            try:
                model = joblib.load(uploaded_model)
                st.success("模型加载成功!")
            except:
                try:
                    model = pickle.load(uploaded_model)
                    st.success("模型加载成功!")
                except Exception as e:
                    st.error(f"模型加载失败: {e}")
    else:
        # 尝试加载默认模型
        try:
            model = joblib.load('lightGBMnew.pkl')
            st.success("默认模型加载成功!")
        except:
            try:
                with open('lightGBMnew.pkl', 'rb') as f:
                    model = pickle.load(f)
                st.success("默认模型加载成功!")
            except Exception as e:
                st.error(f"默认模型加载失败: {e}")
                model = None

    st.markdown("---")
    st.markdown("### 数据管理")

    # 清除历史数据
    if st.button("清除预测历史"):
        st.session_state.prediction_history = pd.DataFrame()
        st.success("预测历史已清除!")

    # 显示历史记录数量
    if not st.session_state.prediction_history.empty:
        st.info(f"当前有 {len(st.session_state.prediction_history)} 条预测记录")

    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("""
    1. **单条预测**: 在主页手工输入数据
    2. **批量预测**: 上传CSV文件进行批量预测
    3. **结果下载**: 预测后可下载结果CSV文件
    """)

# 主页面
st.markdown('<h2 class="sub-header">🔬 患者数据输入</h2>', unsafe_allow_html=True)

# 创建选项卡：单条预测和批量预测
tab1, tab2 = st.tabs(["📝 单条预测", "📁 批量预测"])

# 选项卡1：单条预测
with tab1:
    st.markdown("### 请输入患者数据")

    # 手工输入表格
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        S1 = st.number_input('AFP', min_value=0.00, max_value=100000.00, value=45.00, step=0.01,
                             help="AFP，范围：0.00-100000.00")
        S2 = st.number_input('PIVKA', min_value=0.00, max_value=200000.00, value=100.00, step=0.01,
                             help="PIVKA，范围：0.00-200000.00")
        S3 = st.number_input('GGT', min_value=0.00, max_value=2000.00, value=80.00, step=0.01,
                             help="GGT，范围：0.00-2000.00")
        S4 = st.number_input('HBsAb', min_value=0.00, max_value=2000.00, value=20.00, step=0.01,
                             help="HBsAb，范围：0.00-2000.00")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        S5 = st.number_input('HBeAb', min_value=0.00, max_value=100.00, value=1.00, step=0.01,
                             help="HBeAb，范围：0.00-100.00")
        S6 = st.number_input('HBcAb', min_value=0.00, max_value=100.00, value=1.00, step=0.01,
                             help="HBcAb，范围：0.00-100.00")
        S7 = st.number_input('PT', min_value=0.00, max_value=100.00, value=0.01, step=0.01,
                             help="PT，范围：0.00-100.00")
        st.markdown('</div>', unsafe_allow_html=True)

    # 创建数据字典
    data = {
        'AFP': S1,
        'PIVKA': S2,
        'GGT': S3,
        'HBsAb': S4,
        'HBeAb': S5,
        'HBcAb': S6,
        'PT': S7



    }

    # 显示输入的数据
    st.markdown("### 输入数据预览")
    input_df = pd.DataFrame(data, index=[0])
    st.dataframe(input_df, use_container_width=True)

    # 预测按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 开始预测", use_container_width=True)

    if predict_button and 'model' in locals() and model is not None:
        try:
            # 执行预测
            with st.spinner("正在预测中..."):
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)

            # 显示结果
            st.markdown("### 预测结果")

            # 确定类别标签
            if hasattr(model, 'classes_'):
                class_labels = model.classes_
            else:
                class_labels = [0, 1]  # 默认二分类

            # 显示预测结果
            result_container = st.container()
            with result_container:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

                # 显示预测类别
                st.markdown(f"#### 预测类别: **{prediction[0]}**")

                # 显示概率
                st.markdown("#### 类别概率:")

                for i, prob in enumerate(prediction_proba[0]):
                    label = f"类别 {class_labels[i]}" if i < len(class_labels) else f"类别 {i}"
                    prob_percent = prob * 100

                    # 创建进度条
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**{label}**: {prob_percent:.2f}%")
                    with col2:
                        st.progress(float(prob))

                # 显示阳性概率（假设类别1为阳性）
                if len(prediction_proba[0]) > 1:
                    positive_prob = prediction_proba[0][1] * 100
                    st.markdown(f"#### 阳性概率: **{positive_prob:.2f}%**")

                    # 添加解释性文本
                    if positive_prob > 70:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("**高风险**: 建议进一步检查")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif positive_prob > 30:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("**中等风险**: 建议定期监测")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("**低风险**: 状况良好")
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # 保存到历史记录
            history_entry = {
                '时间戳': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                '预测结果': prediction[0],
                '阳性概率': f"{positive_prob:.2f}%" if len(prediction_proba[0]) > 1 else "N/A",
                **data
            }

            history_df = pd.DataFrame([history_entry])
            st.session_state.prediction_history = pd.concat(
                [st.session_state.prediction_history, history_df],
                ignore_index=True
            )

            st.success("✅ 预测完成！结果已保存到历史记录。")

            # 尝试SHAP解释（如果可用）
            try:
                if hasattr(model, 'predict_proba'):
                    st.markdown("### 特征重要性分析")
                    with st.spinner("正在生成SHAP解释..."):
                        # 创建SHAP解释器
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(input_df)

                        # 绘制SHAP图 - 确保使用中文字体
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # 获取特征名称（使用中文）
                        feature_names = list(input_df.columns)

                        # 绘制SHAP条形图
                        if isinstance(shap_values, list):
                            # 对于多分类问题，取第一个类别的SHAP值
                            shap_df = pd.DataFrame({
                                'features': feature_names,
                                'shap_values': shap_values[0][0] if len(shap_values) > 0 else shap_values[0]
                            })
                        else:
                            # 对于二分类问题
                            shap_df = pd.DataFrame({
                                'features': feature_names,
                                'shap_values': shap_values[0]
                            })

                        # 按绝对值排序
                        shap_df['abs_shap'] = np.abs(shap_df['shap_values'])
                        shap_df = shap_df.sort_values('abs_shap', ascending=True)

                        # 绘制条形图
                        bars = ax.barh(shap_df['features'], shap_df['shap_values'])

                        # 根据值设置颜色
                        for bar in bars:
                            if bar.get_width() >= 0:
                                bar.set_color('#ff6b6b')  # 红色表示正向影响
                            else:
                                bar.set_color('#4d96ff')  # 蓝色表示负向影响

                        # 设置图表属性
                        ax.set_xlabel('SHAP值 (特征影响力)', fontsize=12)
                        ax.set_title('特征重要性分析', fontsize=14, fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)

                        # 添加数值标签
                        for i, (value, feature) in enumerate(zip(shap_df['shap_values'], shap_df['features'])):
                            if value >= 0:
                                ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
                            else:
                                ax.text(value - 0.01, i, f'{value:.3f}', ha='right', va='center', fontsize=10)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # 添加解释说明
                        st.markdown("""
                        **SHAP解释说明:**
                        - **红色条形**: 特征值增加会提高预测概率
                        - **蓝色条形**: 特征值增加会降低预测概率
                        - **条形长度**: 表示特征对预测结果的影响力大小
                        """)
            except Exception as shap_error:
                st.info(f"SHAP解释生成失败: {shap_error}")
                # 尝试替代方法：特征重要性图
                if hasattr(model, 'feature_importances_'):
                    st.markdown("### 特征重要性分析")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # 获取特征重要性
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    # 绘制特征重要性图
                    ax.bar(range(len(importances)), importances[indices])
                    ax.set_xlabel('特征排名', fontsize=12)
                    ax.set_ylabel('重要性', fontsize=12)
                    ax.set_title('特征重要性排序', fontsize=14, fontweight='bold')
                    ax.set_xticks(range(len(importances)))
                    ax.set_xticklabels([input_df.columns[i] for i in indices], rotation=45, ha='right')

                    plt.tight_layout()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")
    elif predict_button:
        st.error("❌ 模型未加载，请先加载模型！")

# 选项卡2：批量预测
with tab2:
    st.markdown("### 批量数据预测")

    # 文件上传
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv', 'xlsx'],
                                     help="请上传包含患者数据的CSV或Excel文件")

    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                batch_data = pd.read_csv(uploaded_file)
            else:
                batch_data = pd.read_excel(uploaded_file)

            st.success(f"成功读取文件，共 {len(batch_data)} 条记录")

            # 显示数据预览
            st.markdown("#### 数据预览")
            st.dataframe(batch_data.head(), use_container_width=True)

            # 检查必要的列
            required_columns = ['AFP', 'PIVKA', 'GGT', 'HBsAb',
                                'HBeAb', 'HBcAb', 'PT'

                                ]

            missing_columns = [col for col in required_columns if col not in batch_data.columns]

            if missing_columns:
                st.error(f"文件缺少以下必要列: {', '.join(missing_columns)}")
                st.info("请确保文件包含以下列: " + ", ".join(required_columns))
            else:
                # 确保数据格式正确
                batch_data = batch_data[required_columns].copy()

                # 批量预测按钮
                if st.button("📊 执行批量预测", use_container_width=True):
                    if 'model' in locals() and model is not None:
                        with st.spinner(f"正在对 {len(batch_data)} 条记录进行预测..."):
                            try:
                                # 执行批量预测
                                predictions = model.predict(batch_data)
                                prediction_probas = model.predict_proba(batch_data)

                                # 创建结果数据框
                                results_df = batch_data.copy()
                                results_df['预测结果'] = predictions

                                # 添加概率列
                                if prediction_probas.shape[1] > 1:
                                    results_df['阴性概率(%)'] = (prediction_probas[:, 0] * 100).round(2)
                                    results_df['阳性概率(%)'] = (prediction_probas[:, 1] * 100).round(2)

                                results_df['预测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                # 保存结果到session state
                                st.session_state.batch_results = results_df

                                # 显示结果
                                st.markdown("#### 批量预测结果")
                                st.dataframe(results_df, use_container_width=True)

                                # 统计信息
                                st.markdown("#### 预测统计")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    total_count = len(results_df)
                                    st.metric("总记录数", total_count)

                                with col2:
                                    positive_count = sum(
                                        results_df['预测结果'] == 1) if '预测结果' in results_df.columns else 0
                                    st.metric("阳性预测数", positive_count)

                                with col3:
                                    negative_count = total_count - positive_count
                                    st.metric("阴性预测数", negative_count)

                                with col4:
                                    if total_count > 0:
                                        positive_rate = (positive_count / total_count) * 100
                                        st.metric("阳性率", f"{positive_rate:.1f}%")

                                # 绘制预测结果分布图
                                st.markdown("#### 预测结果分布")
                                fig, ax = plt.subplots(figsize=(8, 6))

                                if '预测结果' in results_df.columns:
                                    result_counts = results_df['预测结果'].value_counts().sort_index()
                                    labels = ['阴性', '阳性'] if len(result_counts) == 2 else [f'类别{i}' for i in
                                                                                               result_counts.index]

                                    bars = ax.bar(labels, result_counts.values)

                                    # 设置颜色
                                    for i, bar in enumerate(bars):
                                        if i == 0:
                                            bar.set_color('#4CAF50')  # 绿色表示阴性
                                        else:
                                            bar.set_color('#F44336')  # 红色表示阳性

                                    # 添加数值标签
                                    for bar, count in zip(bars, result_counts.values):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                                f'{count}', ha='center', va='bottom')

                                    ax.set_xlabel('预测结果', fontsize=12)
                                    ax.set_ylabel('数量', fontsize=12)
                                    ax.set_title('预测结果分布', fontsize=14, fontweight='bold')
                                    ax.grid(axis='y', alpha=0.3)

                                plt.tight_layout()
                                st.pyplot(fig)

                                st.success(f"✅ 批量预测完成！共处理 {len(results_df)} 条记录。")

                            except Exception as e:
                                st.error(f"批量预测失败: {e}")
                    else:
                        st.error("❌ 模型未加载，请先加载模型！")
        except Exception as e:
            st.error(f"文件读取失败: {e}")

# 结果下载功能
st.markdown('<h2 class="sub-header">💾 结果下载</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # 下载单条预测历史
    if not st.session_state.prediction_history.empty:
        st.markdown("#### 单条预测历史")
        st.dataframe(st.session_state.prediction_history, use_container_width=True)

        # 转换为CSV
        csv = st.session_state.prediction_history.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 下载预测历史(CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

with col2:
    # 下载批量预测结果
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        st.markdown("#### 批量预测结果")

        # 转换为CSV
        csv = st.session_state.batch_results.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 下载批量预测结果(CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # 也提供Excel格式
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            st.session_state.batch_results.to_excel(writer, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">📥 下载批量预测结果(Excel)</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("暂无批量预测结果可供下载")

# 历史数据分析（如果数据足够多）
if not st.session_state.prediction_history.empty and len(st.session_state.prediction_history) > 1:
    st.markdown('<h2 class="sub-header">📊 历史数据分析</h2>', unsafe_allow_html=True)

    # 创建图表
    try:
        history_df = st.session_state.prediction_history.copy()

        # 转换概率为数值
        if '阳性概率' in history_df.columns:
            history_df['阳性概率数值'] = history_df['阳性概率'].str.replace('%', '').astype(float)

        # 创建图表
        col1, col2 = st.columns(2)

        with col1:
            # 预测结果分布
            if '预测结果' in history_df.columns:
                result_counts = history_df['预测结果'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(8, 6))

                # 为饼图创建标签
                if len(result_counts) == 2:
                    labels = ['阴性', '阳性']
                else:
                    labels = [f'类别{idx}' for idx in result_counts.index]

                wedges, texts, autotexts = ax1.pie(result_counts.values, labels=labels, autopct='%1.1f%%',
                                                   startangle=90, colors=['#4CAF50', '#F44336'])

                # 设置文本属性
                for text in texts:
                    text.set_fontsize(12)
                for autotext in autotexts:
                    autotext.set_fontsize(11)
                    autotext.set_color('white')

                ax1.set_title('预测结果分布', fontsize=14, fontweight='bold')
                ax1.axis('equal')  # 确保饼图是圆形
                st.pyplot(fig1)

        with col2:
            # 阳性概率分布
            if '阳性概率数值' in history_df.columns:
                fig2, ax2 = plt.subplots(figsize=(8, 6))

                # 创建直方图
                n, bins, patches = ax2.hist(history_df['阳性概率数值'], bins=20, edgecolor='black', color='#2196F3')

                # 添加平均值线
                mean_value = history_df['阳性概率数值'].mean()
                ax2.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
                ax2.text(mean_value + 1, max(n) * 0.9, f'平均: {mean_value:.1f}%', color='red')

                ax2.set_xlabel('阳性概率 (%)', fontsize=12)
                ax2.set_ylabel('频数', fontsize=12)
                ax2.set_title('阳性概率分布', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)

                st.pyplot(fig2)

    except Exception as e:
        st.warning(f"数据分析时出现错误: {e}")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p>医疗预测分析系统（测试用）by Zoldyck | 版本 1.0 | 最后更新: 2026/03/09</p>
</div>
""", unsafe_allow_html=True)