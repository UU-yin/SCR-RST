# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:51:46 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from scipy import stats

# 设置页面
st.set_page_config(
    page_title="统计分析工具",
    page_icon="📊",
    layout="wide"
)

# 标题和说明
st.title("📊 统计分析工具")
st.markdown("""
提供多种稳健统计分析方法，用于处理包含异常值的数据集。
支持迭代稳健统计法、四分位稳健统计法和Q/Hampel法。
""")

# =============================================
# 新增：数据验证和错误处理模块
# =============================================

class DataValidator:
    """数据验证器类"""
    
    @staticmethod
    def validate_numeric_string(data_string):
        """
        验证数值字符串格式
        返回: (is_valid, cleaned_data, error_message)
        """
        if not data_string or data_string.strip() == "":
            return False, None, "输入数据不能为空"
        
        # 清理和分割数据
        lines = data_string.strip().split('\n')
        data_list = []
        
        for line in lines:
            # 支持逗号、空格、分号分隔
            items = re.split(r'[,;\s]+', line.strip())
            for item in items:
                if item:  # 跳过空字符串
                    try:
                        # 尝试转换为浮点数
                        value = float(item)
                        data_list.append(value)
                    except ValueError:
                        return False, None, f"数据格式错误: '{item}' 不是有效的数字"
        
        if len(data_list) == 0:
            return False, None, "未找到有效的数值数据"
        
        if len(data_list) < 3:
            return False, None, "数据点数量不足，至少需要3个数据点进行分析"
        
        return True, np.array(data_list), "数据格式验证通过"
    
    @staticmethod
    def validate_data_range(data_array, min_value=-1e10, max_value=1e10):
        """
        验证数据范围
        """
        if np.any(data_array < min_value):
            return False, f"发现过小数值: {np.min(data_array):.4f} < {min_value}"
        if np.any(data_array > max_value):
            return False, f"发现过大数值: {np.max(data_array):.4f} > {max_value}"
        
        return True, "数据范围验证通过"
    
    @staticmethod
    def validate_data_variance(data_array):
        """
        验证数据方差，避免所有数据相同
        """
        if np.std(data_array) == 0:
            return False, "所有数据值相同，无法进行统计分析"
        return True, "数据方差验证通过"
    
    @staticmethod
    def detect_potential_outliers(data_array, method='iqr', threshold=3):
        """
        检测潜在异常值
        """
        if len(data_array) < 5:
            return [], "数据点不足，无法进行异常值检测"
        
        outliers_info = []
        
        if method == 'iqr':
            Q1 = np.percentile(data_array, 25)
            Q3 = np.percentile(data_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
            if len(outliers) > 0:
                outliers_info.append(f"IQR方法检测到 {len(outliers)} 个潜在异常值")
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_array))
            outliers = data_array[z_scores > threshold]
            if len(outliers) > 0:
                outliers_info.append(f"Z分数方法检测到 {len(outliers)} 个潜在异常值 (阈值: {threshold})")
        
        return outliers, outliers_info
    
    @staticmethod
    def comprehensive_validation(data_string):
        """
        综合数据验证
        返回: (is_valid, data_array, validation_report)
        """
        validation_report = []
        
        # 1. 格式验证
        is_valid, data_array, error_msg = DataValidator.validate_numeric_string(data_string)
        if not is_valid:
            return False, None, [f"❌ 格式验证失败: {error_msg}"]
        validation_report.append(f"✅ {error_msg}")
        
        # 2. 数据范围验证
        is_valid, range_msg = DataValidator.validate_data_range(data_array)
        if not is_valid:
            return False, None, validation_report + [f"❌ 范围验证失败: {range_msg}"]
        validation_report.append(f"✅ {range_msg}")
        
        # 3. 方差验证
        is_valid, variance_msg = DataValidator.validate_data_variance(data_array)
        if not is_valid:
            return False, None, validation_report + [f"❌ 方差验证失败: {variance_msg}"]
        validation_report.append(f"✅ {variance_msg}")
        
        # 4. 异常值检测
        outliers, outliers_info = DataValidator.detect_potential_outliers(data_array)
        if outliers_info:
            validation_report.append(f"⚠️ {outliers_info[0]}")
            if len(outliers) > 0:
                validation_report.append(f"   异常值: {', '.join([f'{x:.4f}' for x in sorted(outliers)])}")
        else:
            validation_report.append("✅ 未发现明显异常值")
        
        # 5. 数据统计信息
        validation_report.extend([
            f"📊 数据统计摘要:",
            f"   数据点数: {len(data_array)}",
            f"   数据范围: [{np.min(data_array):.4f}, {np.max(data_array):.4f}]",
            f"   平均值: {np.mean(data_array):.4f}",
            f"   标准差: {np.std(data_array, ddof=1):.4f}"
        ])
        
        return True, data_array, validation_report

# 侧边栏 - 参数设置和方法选择
st.sidebar.header("⚙️ 分析设置")

# 方法选择
method = st.sidebar.selectbox(
    "选择统计方法:",
    ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法"],
    help="选择适合数据特征的稳健统计方法"
)

# 根据选择的方法显示相应参数
if method == "迭代稳健统计法":
    k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
    max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)
elif method == "四分位稳健统计法":
    st.sidebar.info("四分位法使用固定参数计算")
elif method == "Q/Hampel法":
    st.sidebar.info("Q/Hampel法使用标准参数计算")

# 数据输入方式选择
input_method = st.radio("数据输入方式:", 
                       ["手动输入", "文件上传", "示例数据"])
data = None

if input_method == "手动输入":
    st.subheader("📝 手动输入数据")
    
    # 初始化所有必要的会话状态
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6"
    
    # 关键修复：确保历史记录正确初始化
    if 'data_history' not in st.session_state:
        # 只保存初始状态，不重复保存
        st.session_state.data_history = []
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
        
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    # 创建文本输入框，其key依赖于reset_counter
    current_data = st.text_area(
        "请输入数据（每行一个数值或用逗号分隔）:",
        value=st.session_state.manual_data,
        height=150,
        key=f"manual_input_{st.session_state.reset_counter}"
    )

    # 更新session_state中的数据 - 关键修复
    if current_data != st.session_state.manual_data:
        # 只有当数据真正变化且不是空字符串时才保存到历史记录
        if st.session_state.manual_data and current_data != st.session_state.manual_data:
            # 保存当前状态到历史记录
            st.session_state.data_history.append(st.session_state.manual_data)
            # 限制历史记录长度，避免内存问题
            if len(st.session_state.data_history) > 10:
                st.session_state.data_history = st.session_state.data_history[-10:]
        
        # 更新当前数据
        st.session_state.manual_data = current_data

    # 创建操作按钮
    col1, col2, col3 = st.columns([2, 1, 1])

    def clear_data():
        """一键清除数据的回调函数"""
        # 保存当前状态到历史记录（只有在有内容时）
        if st.session_state.manual_data and st.session_state.manual_data.strip():
            st.session_state.data_history.append(st.session_state.manual_data)
        
        st.session_state.manual_data = ""
        st.session_state.data_loaded = False
        st.session_state.processed_data = None
        st.session_state.reset_counter += 1  # 改变计数器以重置文本区域

    def undo_data():
        """撤销操作的回调函数"""
        # 关键修复：检查历史记录是否为空
        if st.session_state.data_history:
            # 从历史记录中获取上一个状态
            previous_data = st.session_state.data_history.pop()
            st.session_state.manual_data = previous_data
            st.session_state.data_loaded = False
            st.session_state.processed_data = None
            st.session_state.reset_counter += 1  # 改变计数器以重置文本区域
        else:
            # 如果没有历史记录，至少重置计数器以刷新界面
            st.session_state.reset_counter += 1

    def analyze_data():
        """分析数据的回调函数"""
        try:
            # 使用新的数据验证器
            is_valid, validated_data, validation_report = DataValidator.comprehensive_validation(
                st.session_state.manual_data
            )
            
            if is_valid:
                st.session_state.processed_data = validated_data
                st.session_state.data_loaded = True
                st.session_state.validation_report = validation_report
                st.success(f"✅ 数据验证通过！成功解析 {len(validated_data)} 个数据点")
                
                # 显示详细的验证报告
                with st.expander("📋 查看详细验证报告", expanded=True):
                    for line in validation_report:
                        if line.startswith("❌"):
                            st.error(line)
                        elif line.startswith("⚠️"):
                            st.warning(line)
                        elif line.startswith("📊"):
                            st.write("**" + line + "**")
                        else:
                            st.write(line)
            else:
                st.error("❌ 数据验证失败")
                with st.expander("📋 查看验证详情", expanded=True):
                    for line in validation_report:
                        if line.startswith("❌"):
                            st.error(line)
                        else:
                            st.write(line)
                
        except Exception as e:
            st.error(f"❌ 分析过程中发生错误: {str(e)}")
            st.info("💡 建议检查数据格式，确保所有输入都是有效的数字")

    with col1:
        st.button("分析数据", 
                  use_container_width=True, 
                  type="primary",
                  on_click=analyze_data)

    with col2:
        st.button("一键清除", 
                  use_container_width=True, 
                  type="secondary",
                  help="清空所有数据",
                  on_click=clear_data)

    with col3:
        undo_disabled = len(st.session_state.data_history) == 0
        st.button("↶ 撤销", 
                  use_container_width=True, 
                  disabled=undo_disabled,
                  help="恢复到上一次的数据状态",
                  on_click=undo_data)
    
    # 调试信息 - 帮助诊断问题
    with st.expander("调试信息"):
        st.write(f"当前数据: {st.session_state.manual_data}")
        st.write(f"历史记录长度: {len(st.session_state.data_history)}")
        st.write(f"历史记录内容: {st.session_state.data_history}")
        st.write(f"重置计数器: {st.session_state.reset_counter}")

    # 将处理后的数据传递给应用的其余部分
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        data = st.session_state.processed_data

elif input_method == "文件上传":
    st.subheader("📁 上传数据文件")
    
    # 首先放置文件上传器
    uploaded_file = st.file_uploader("选择CSV或TXT文件", type=['csv', 'txt'])
    
    # 然后在下方显示格式说明和示例
    with st.expander("📝 查看文件格式说明和示例", expanded=False):
        st.markdown("""
        **TXT文件格式要求：**
        - 每行一个数值
        - 支持整数和小数
        - 空行会自动忽略
        
        **CSV文件格式要求：**
        - 第一列包含数值数据
        - 可以有表头，也可以没有
        
        **示例文件内容：**
        ```
        54.4
        54.6
        54.2
        54.3
        53.9
        ```
        """)
        
        # 提供示例文件下载
        example_content = "54.4\n54.6\n54.2\n54.3\n53.9"
        st.download_button(
            label="下载示例TXT文件",
            data=example_content,
            file_name="example_data.txt",
            mime="text/plain",
            help="点击下载示例TXT文件，了解正确的数据格式"
        )
    
    if uploaded_file is not None:
        try:
            file_content = ""
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # 假设第一列是数据
                data_values = df.iloc[:, 0].values
                # 将数据转换为字符串格式进行验证
                file_content = "\n".join([str(x) for x in data_values])
            else:
                # 文本文件，每行一个数字
                file_content = uploaded_file.read().decode()
            
            # 使用数据验证器验证文件数据
            is_valid, validated_data, validation_report = DataValidator.comprehensive_validation(file_content)
            
            if is_valid:
                data = validated_data
                st.success(f"✅ 文件验证通过！成功加载 {len(data)} 个数据点")
                
                # 显示验证报告
                with st.expander("📋 查看文件验证报告", expanded=True):
                    for line in validation_report:
                        if line.startswith("❌"):
                            st.error(line)
                        elif line.startswith("⚠️"):
                            st.warning(line)
                        elif line.startswith("📊"):
                            st.write("**" + line + "**")
                        else:
                            st.write(line)
                
                st.write("前10个数据:", data[:10])
            else:
                st.error("❌ 文件数据验证失败")
                with st.expander("📋 查看验证详情", expanded=True):
                    for line in validation_report:
                        if line.startswith("❌"):
                            st.error(line)
                        else:
                            st.write(line)
            
        except Exception as e:
            st.error(f"❌ 文件读取错误: {e}")
            st.info("💡 请确保文件格式正确：每行一个数值，且均为有效数字")

else:  # 示例数据
    st.subheader("🎯 示例数据分析")
    example_data = np.array([
        54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
        54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
        54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
        54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
        54.5, 55.9, 53.2, 54.6
    ])
    
    # 对示例数据进行验证
    example_data_str = ", ".join([str(x) for x in example_data])
    is_valid, validated_data, validation_report = DataValidator.comprehensive_validation(example_data_str)
    
    # 显示示例数据信息
    st.write(f"示例数据已加载，包含 {len(example_data)} 个测量值")
    
    if is_valid:
        st.success("✅ 示例数据验证通过")
    
    # 添加一个可展开的区域显示所有原始数据值
    with st.expander("📋 查看所有示例数据值", expanded=False):
        # 创建数据框显示所有数据
        df_example = pd.DataFrame({
            '数据编号': range(1, len(example_data) + 1),
            '数值': example_data
        })
        st.dataframe(df_example, use_container_width=True)
        
        # 显示验证报告
        st.write("**数据验证报告:**")
        for line in validation_report:
            if line.startswith("❌"):
                st.error(line)
            elif line.startswith("⚠️"):
                st.warning(line)
            elif line.startswith("📊"):
                st.write("**" + line + "**")
            else:
                st.write(line)
        
        # 同时显示基本统计信息
        st.write("**基本统计信息:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均值", f"{np.mean(example_data):.4f}")
        with col2:
            st.metric("标准差", f"{np.std(example_data, ddof=1):.4f}")
        with col3:
            st.metric("最小值", f"{np.min(example_data):.4f}")
        with col4:
            st.metric("最大值", f"{np.max(example_data):.4f}")
    
    # 设置数据变量，以便后续分析
    data = example_data

# 方法描述
st.sidebar.header("📚 方法说明")
if method == "迭代稳健统计法":
    st.sidebar.info("""
    **迭代稳健统计法**（算法A）通过迭代过程逐步修正异常值影响，
    收敛后得到稳健的统计估计。
    """)
elif method == "四分位稳健统计法":
    st.sidebar.info("""
    **四分位稳健统计法**以数据排序为基础，使用数据集中段50%的数据，
    崩溃点为25%，具有易于计算、操作简单的特点。
    """)
else:  # Q/Hampel法
    st.sidebar.info("""
    **Q/Hampel法**结合Q方法计算的稳健标准差和Hampel方法计算的
    稳健平均值，具有较好的抗异常值干扰能力。
    """)

# 统计方法实现
def iterative_robust_algorithm(data, max_iterations=50, k=1.5):
    """
    迭代稳健统计法（原算法A）
    """
    n = len(data)
    
    # 初始值
    X_star = np.median(data)
    abs_deviations = np.abs(data - X_star)
    median_abs_deviation = np.median(abs_deviations)
    S_star = 1.483 * median_abs_deviation
    
    # 迭代过程
    converged = False
    iteration = 0
    history = []
    
    while iteration < max_iterations and not converged:
        iteration += 1
        prev_X_star = X_star
        prev_S_star = S_star
        
        # 计算δ并修正数据点
        delta = k * S_star
        Xj_star = np.where(data < X_star - delta, X_star - delta, 
                          np.where(data > X_star + delta, X_star + delta, data))
        
        # 重新计算
        X_star = np.mean(Xj_star)
        sum_squared_deviations = np.sum((Xj_star - X_star)**2)
        S_star = 1.134 * np.sqrt(sum_squared_deviations / (n-1))
        
        # 记录历史
        history.append({
            'iteration': iteration,
            'X_star': X_star,
            'S_star': S_star,
            'delta': delta
        })
        
        # 检查收敛
        if (int(prev_X_star * 1000) == int(X_star * 1000) and 
            int(prev_S_star * 1000) == int(S_star * 1000)):
            converged = True
    
    # 最终结果
    final_delta = k * S_star
    lower_limit = X_star - final_delta
    upper_limit = X_star + final_delta
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    Z_scores = (data - X_star) / S_star
    
    return {
        'robust_mean': X_star,
        'robust_std': S_star,
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'iterations': iteration,
        'converged': converged,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'history': history,
        'method_name': '迭代稳健统计法'
    }

def quartile_robust_algorithm(data):
    """
    四分位稳健统计法
    """
    # 数据排序
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # 计算中位值
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    # 计算四分位数
    q1 = np.percentile(data, 25)  # 下四分位数
    q3 = np.percentile(data, 75)  # 上四分位数
    
    # 计算四分位距和标准化四分位距
    iqr = q3 - q1
    niqr = 0.7413 * iqr  # 标准化四分位距
    
    # 计算正常值范围（基于四分位数）
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    
    # 识别离群值
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    
    # 计算Z比分数（使用中位值和NIQR）
    Z_scores = (data - median) / niqr
    
    return {
        'robust_mean': median,      # 使用中位值作为稳健平均值
        'robust_std': niqr,         # 使用NIQR作为稳健标准差
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'niqr': niqr,
        'method_name': '四分位稳健统计法',
        'lower_limit': lower_limit,
        'upper_limit': upper_limit
    }

def q_hampel_robust_algorithm(data):
    """
    Q/Hampel稳健统计方法
    """
    # 简化版的Q/Hampel实现
    # 注意：完整的Q/Hampel方法需要多个实验室数据，这里提供简化版本
    
    n = len(data)
    
    # 计算中位值（作为Hampel方法的初始估计）
    median = np.median(data)
    
    # 计算Q方法的稳健标准差（简化版）
    # 基于成对绝对差的中位数
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append(abs(data[i] - data[j]))
    
    if len(pairs) > 0:
        q_std = np.median(pairs) / 1.0484  # 调整系数
    else:
        q_std = np.std(data, ddof=1)
    
    # Hampel方法的稳健平均值（迭代加权法简化版）
    # 使用中位值作为初始估计
    current_mean = median
    max_iterations = 10
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        # 计算残差
        residuals = data - current_mean
        mad = np.median(np.abs(residuals))
        
        if mad == 0:
            break
            
        # 标准化残差
        standardized_residuals = residuals / (1.4826 * mad)
        
        # Hampel权重函数
        weights = np.ones_like(data)
        mask1 = np.abs(standardized_residuals) > 1.5
        mask2 = np.abs(standardized_residuals) > 3
        mask3 = np.abs(standardized_residuals) > 4.5
        
        weights[mask1] = 1.5 / np.abs(standardized_residuals[mask1])
        weights[mask2] = 0
        weights[mask3] = 0
        
        # 更新均值
        new_mean = np.sum(weights * data) / np.sum(weights)
        
        # 检查收敛
        if abs(new_mean - current_mean) < tolerance:
            break
            
        current_mean = new_mean
    
    # 计算正常值范围
    lower_limit = current_mean - 3 * q_std
    upper_limit = current_mean + 3 * q_std
    
    # 识别离群值
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    
    # 计算Z比分数
    Z_scores = (data - current_mean) / q_std
    
    return {
        'robust_mean': current_mean,  # Hampel稳健平均值
        'robust_std': q_std,          # Q方法稳健标准差
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'method_name': 'Q/Hampel法',
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'weights': weights if 'weights' in locals() else np.ones_like(data)
    }

# 执行分析
if data is not None and len(data) > 0:
    try:
        st.markdown("---")
        st.subheader(f"📈 {method}分析结果")
        
        # 新增：数据分布可视化
        st.subheader("输入数据正态分布分析")
        
        # 创建两列布局，左侧显示统计信息，右侧显示分布图
        dist_col1, dist_col2 = st.columns([1, 2])
        
        with dist_col1:
            st.write("**数据统计摘要:**")
            st.write(f"数据点数: {len(data)}")
            st.write(f"平均值: {np.mean(data):.4f}")
            st.write(f"标准差: {np.std(data, ddof=1):.4f}")
            st.write(f"最小值: {np.min(data):.4f}")
            st.write(f"最大值: {np.max(data):.4f}")
            st.write(f"中位数: {np.median(data):.4f}")
            
            # 正态性检验 - 放在统计信息下方，图表上方
            from scipy.stats import shapiro
            if len(data) >= 3 and len(data) <= 5000:  # Shapiro-Wilk检验的适用范围
                stat, p_value = shapiro(data)
                
                # 在同一列中上下排列，使用颜色编码
                st.write(f"正态性检验p值: {p_value:.4f}")
                if p_value > 0.05:
                    st.write(":green[数据符合正态分布 (p > 0.05)]")
                else:
                    st.write(":red[数据可能不符合正态分布 (p ≤ 0.05)]")
        
        with dist_col2:
            # 创建数据分布图
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            
            # 绘制直方图
            n, bins, patches = ax_dist.hist(data, bins=15, alpha=0.7, color='skyblue', 
                                           edgecolor='black', density=True, label='Data Distribution')
            
            # 绘制正态分布曲线
            from scipy.stats import norm
            xmin, xmax = ax_dist.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, np.mean(data), np.std(data, ddof=1))
            ax_dist.plot(x, p, 'k', linewidth=2, label='Normal Distribution Curve')
            
            # 设置图形属性
            ax_dist.set_title('Normal‐Probability Benchmarking of Input Data', fontsize=14, fontweight='bold')
            ax_dist.set_xlabel('Data Value', fontsize=12)
            ax_dist.set_ylabel('Probability Density', fontsize=12)
            ax_dist.legend()
            ax_dist.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_dist)
        
        # 执行稳健统计分析
        with st.spinner(f"正在执行{method}分析..."):
            try:
                if method == "迭代稳健统计法":
                    results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value)
                elif method == "四分位稳健统计法":
                    results = quartile_robust_algorithm(data)
                else:  # Q/Hampel法
                    results = q_hampel_robust_algorithm(data)
                
                # 创建两列布局
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
                    st.metric("稳健标准差", f"{results['robust_std']:.6f}")
                    
                with col2:
                    if 'iterations' in results:
                        st.metric("迭代次数", results['iterations'])
                    st.metric("离群值数量", len(results['outliers']))
                
                # 方法特定结果显示
                if method == "四分位稳健统计法":
                    st.info("📊 **四分位统计量:**")
                    col3, col4, col5, col6 = st.columns(4)
                    with col3:
                        st.metric("下四分位数(Q1)", f"{results['q1']:.6f}")
                    with col4:
                        st.metric("上四分位数(Q3)", f"{results['q3']:.6f}")
                    with col5:
                        st.metric("四分位距(IQR)", f"{results['iqr']:.6f}")
                    with col6:
                        st.metric("标准化四分位距(NIQR)", f"{results['niqr']:.6f}")
                
                # 详细结果
                st.subheader("📋 详细结果")
                
                st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
                if 'converged' in results:
                    st.write(f"**收敛状态**: {'是' if results['converged'] else '否'}")
                
                if len(results['outliers']) > 0:
                    # 将np.float64转换为Python原生float类型
                    outliers_list = [float(x) for x in sorted(results['outliers'])]
                    st.write(f"**离群值**: {outliers_list}")
                else:
                    st.write("**离群值**: 无")
                
                # Z比分数统计
                z_scores_abs = np.abs(results['Z_scores'])
                satisfactory = np.sum(z_scores_abs <= 2)
                questionable = np.sum((z_scores_abs > 2) & (z_scores_abs <= 3))
                unsatisfactory = np.sum(z_scores_abs > 3)
                
                st.write("**Z比分数分类**:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("满意 (|Z| ≤ 2)", f"{satisfactory} 个")
                with col2:
                    st.metric("可疑 (2 < |Z| ≤ 3)", f"{questionable} 个")
                with col3:
                    st.metric("不满意 (|Z| > 3)", f"{unsatisfactory} 个")
                
                # 可视化 - 使用新的Z值柱状图
                st.subheader("数据可视化")
                
                # 创建数据框用于可视化
                df_clean = pd.DataFrame({
                    'Original_Data': data,
                    'Z_Score': results['Z_scores']
                })
                
                # 根据Z值进行分类
                def classify_data(row):
                    if abs(row['Z_Score']) <= 2:
                        return 'Satisfactory'
                    elif 2 < abs(row['Z_Score']) <= 3:
                        return 'Questionable'
                    else:
                        return 'Unsatisfactory'
                
                df_clean['Category'] = df_clean.apply(classify_data, axis=1)
                
                # 按照Z值从大到小排序
                df_sorted = df_clean.sort_values('Z_Score', ascending=False)
                
                # 创建Z值柱状图
                fig, ax = plt.subplots(figsize=(14, 16))
                
                # 设置类别对应的高饱和度颜色
                color_map = {
                    'Satisfactory': '#00FF00',    # 高饱和度绿色
                    'Questionable': '#FFA500',    # 高饱和度橙色
                    'Unsatisfactory': '#FF0000'    # 高饱和度红色
                }
                
                # 创建一个统一颜色的列表
                colors = [color_map[cat] for cat in df_sorted['Category']]
                
                # 绘制所有数据点的柱状图，按Z值排序
                y_positions = range(len(df_sorted))
                bars = ax.barh(y_positions, 
                               df_sorted['Z_Score'], 
                               color=colors, 
                               alpha=0.6,  # 降低透明度使颜色更柔和
                               height=0.8,
                               edgecolor='white',  # 使用白色边框使柱状图更清晰
                               linewidth=0.5)
                
                # 在柱状图上标注Z值
                for i, (bar, z_value) in enumerate(zip(bars, df_sorted['Z_Score'])):
                    # 使用黑色文字确保在较淡的背景上可读
                    text_color = 'black'
                    ax.text(bar.get_width() + 0.05 * (1 if bar.get_width() >= 0 else -1), 
                            bar.get_y() + bar.get_height()/2, 
                            f'{z_value:.2f}', 
                            ha='left' if bar.get_width() >= 0 else 'right', 
                            va='center', fontsize=9, fontweight='bold',
                            color=text_color)
                
                # 设置图形属性
                ax.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
                ax.set_ylabel('Original Data ID', fontsize=14, fontweight='bold')
                ax.set_title('Z-Score Distribution (Sorted)', fontsize=18, fontweight='bold', pad=40)
                
                # 添加图例 - 放在图表上方，标题下方
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=color_map['Satisfactory'], alpha=0.6, label='Satisfactory (|Z| ≤ 2)'),
                    Patch(facecolor=color_map['Questionable'], alpha=0.6, label='Questionable (2 < |Z| ≤ 3)'),
                    Patch(facecolor=color_map['Unsatisfactory'], alpha=0.6, label='Unsatisfactory (|Z| > 3)')
                ]
                
                # 将图例放在图表上方，标题下方
                legend = ax.legend(handles=legend_elements, title='Category', title_fontsize=12, fontsize=11, 
                                  loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=True)
                
                # 设置Y轴刻度 - 使用原始数据编号作为标签
                ax.set_yticks(y_positions)
                ax.set_yticklabels([f"{idx}" for idx in df_sorted.index])
                
                # 添加零线参考线
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # 添加阈值线
                ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
                
                # 添加网格
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # 反转Y轴，使最大的Z值在顶部
                ax.invert_yaxis()
                
                # 设置背景色为白色，使高饱和度颜色更加突出
                ax.set_facecolor('white')
                
                # 调整子图参数，为顶部图例和标题留出更多空间
                plt.subplots_adjust(top=0.88)
                
                # 调整布局
                plt.tight_layout()
                
                # 显示图表
                st.pyplot(fig)
                
                # 添加下载柱状图功能
                st.subheader("💾 下载图表")

                # 创建两列布局放置下载按钮
                col1, col2 = st.columns(2)

                with col1:
                    # 将图表保存为PNG格式并提供下载
                    from io import BytesIO
                    buffer_png = BytesIO()
                    fig.savefig(buffer_png, format="png", dpi=300, bbox_inches="tight")
                    buffer_png.seek(0)
                
                    st.download_button(
                        label="📥 下载PNG格式图表",
                        data=buffer_png,
                        file_name=f"z_score_chart_{method}.png",
                        mime="image/png",
                        help="下载高分辨率PNG格式的Z值分布图"
                    )

                with col2:
                    # 将图表保存为PDF格式并提供下载
                    buffer_pdf = BytesIO()
                    fig.savefig(buffer_pdf, format="pdf", bbox_inches="tight")
                    buffer_pdf.seek(0)
                
                    st.download_button(
                        label="📥 下载PDF格式图表",
                        data=buffer_pdf,
                        file_name=f"z_score_chart_{method}.pdf",
                        mime="application/pdf",
                        help="下载PDF格式的Z值分布图，适合打印和报告"
                    )

                # 添加提示信息
                st.info("💡 提示：PNG格式适合在演示文稿和网页中使用，PDF格式适合打印和学术报告。")
                
                # 导出功能
                st.subheader("💾 导出结果")
                
                # 创建结果DataFrame
                result_df = pd.DataFrame({
                    '原始数据': data,
                    'Z比分数': results['Z_scores'],
                    '分类': np.where(np.abs(results['Z_scores']) <= 2, '满意',
                                   np.where(np.abs(results['Z_scores']) <= 3, '可疑', '不满意'))
                })
                
                # 下载CSV
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="下载完整结果CSV",
                    data=csv,
                    file_name=f"{method}_分析结果.csv",
                    mime="text/csv"
                )
                
                # 下载报告
                report = f"""
{method}分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
数据点数: {len(data)}
使用方法: {method}

关键结果:
--------
稳健平均值: {results['robust_mean']:.6f}
稳健标准差: {results['robust_std']:.6f}
正常值范围: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]
离群值数量: {len(results['outliers'])}

"""
                
                if method == "四分位稳健统计法":
                    report += f"""
四分位统计量:
-----------
下四分位数(Q1): {results['q1']:.6f}
上四分位数(Q3): {results['q3']:.6f}
四分位距(IQR): {results['iqr']:.6f}
标准化四分位距(NIQR): {results['niqr']:.6f}

"""
                
                if 'iterations' in results:
                    report += f"迭代次数: {results['iterations']}\n"
                
                report += f"""
数据质量分类:
-----------
满意 (|Z| ≤ 2): {satisfactory} 个数据点
可疑 (2 < |Z| ≤ 3): {questionable} 个数据点  
不满意 (|Z| > 3): {unsatisfactory} 个数据点

离群值列表:
----------
{', '.join([str(float(x)) for x in results['outliers']])}
"""
                
                st.download_button(
                    label="下载分析报告",
                    data=report,
                    file_name=f"{method}_分析报告.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ 统计分析过程中发生错误: {str(e)}")
                st.info("💡 这可能是因为数据特征不适合所选的分析方法，请尝试其他统计方法或检查数据质量")
                
    except Exception as e:
        st.error(f"❌ 分析流程中出现意外错误: {str(e)}")
        st.info("💡 请尝试重新加载页面或检查输入数据")

else:
    st.info("👆 请先输入或上传数据以开始分析")

# 页脚
st.markdown("---")
st.markdown("""
**方法说明:**
- **迭代稳健统计法**: 通过迭代过程逐步修正异常值影响
- **四分位稳健统计法**: 基于数据排序，使用中段50%数据，崩溃点25%
- **Q/Hampel法**: 结合Q方法稳健标准差和Hampel方法稳健平均值
""")

# 在页面底部添加简化的反馈功能
st.markdown("---")
st.subheader("💬 用户反馈")

# 使用扩展器形式
with st.expander("💬 有问题或建议？点击这里联系我们", expanded=False):
    st.markdown("""
    **技术支持与反馈**
    
    我们重视每一位用户的反馈，如果您遇到以下情况：
    - 使用过程中遇到问题
    - 有功能改进建议
    - 发现数据计算异常
    - 其他任何疑问
    
    请通过以下方式联系我们：
    
    📩 **xxx@163.com**
    
    **联系人**：x博士
       
    感谢您帮助我们变得更好！
    """)