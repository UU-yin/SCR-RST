# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 11:30:10 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import json
from scipy import stats
import matplotlib as mpl
import matplotlib.font_manager as fm

# 设置中文字体
def set_chinese_font():
    """设置中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

# =============================================
# 数据验证器类
# =============================================

class DataValidator:
    """数据验证器类"""
    
    @staticmethod
    def validate_numeric_string_with_blanks(data_string):
        """验证数值字符串格式，支持空白数据"""
        if data_string is None:
            return False, [], [], 0, "输入数据为None", {}

        if not isinstance(data_string, str):
            return False, [], [], 0, f"输入数据类型错误: {type(data_string)}，应为字符串", {}
        
        if not data_string or data_string.strip() == "":
            return False, [], [], 0, "输入数据不能为空", {}
        
        lines = data_string.strip().split('\n')
        original_data = []
        clean_data = []
        blank_positions = []
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': 0
        }
        
        line_num = 0
        max_decimal_places = 0
        
        for line in lines:
            line_num += 1
            items = re.split(r'[,;\s]+', line.strip())
            col_num = 0
            for item in items:
                col_num += 1
                if item and item.strip():
                    try:
                        value = float(item)
                        original_data.append(value)
                        clean_data.append(value)
                        
                        str_value = str(value)
                        if '.' in str_value:
                            decimal_part = str_value.split('.')[1].rstrip('0')
                            decimal_places = len(decimal_part)
                        else:
                            decimal_places = 0
                        
                        max_decimal_places = max(max_decimal_places, decimal_places)
                        
                        decimal_info['decimal_places_count'][decimal_places] = \
                            decimal_info['decimal_places_count'].get(decimal_places, 0) + 1
                        decimal_info['max_decimal_places'] = max_decimal_places
                        
                        if decimal_info.get('previous_decimal_places') is not None and decimal_info['previous_decimal_places'] != decimal_places:
                            decimal_info['consistent_decimals'] = False
                        decimal_info['previous_decimal_places'] = decimal_places
                        
                    except ValueError:
                        return False, [], [], 0, f"数据格式错误: 第{line_num}行 '{item}' 不是有效的数字", {}
                else:
                    original_data.append(None)
                    blank_positions.append((line_num, col_num))
        
        decimal_info['detected_decimal_places'] = max_decimal_places
        blank_count = len(blank_positions)
        return True, original_data, clean_data, blank_count, "数据格式验证通过", decimal_info
    
    @staticmethod
    def comprehensive_validation(data_string, calculation_scheme="规范展示方案"):
        """综合数据验证"""
        validation_report = []
        blank_count = 0
        
        is_valid, original_data, clean_data, blank_count, error_msg, decimal_info = \
            DataValidator.validate_numeric_string_with_blanks(data_string)
        
        if not is_valid:
            return False, [], [], blank_count, [f"❌ 格式验证失败: {error_msg}"], decimal_info
        
        validation_report.append(f"✅ {error_msg}")
        
        if len(clean_data) == 0:
            return False, original_data, [], blank_count, validation_report + ["❌ 没有有效数据"], decimal_info
        
        validation_report.extend([
            f"📊 数据统计摘要:",
            f"   总数据点数: {len(original_data)}",
            f"   实际可分析数据数: {len(clean_data)}",
            f"   空白数据数: {blank_count}",
            f"   有效数据范围: [{np.min(clean_data):.4f}, {np.max(clean_data):.4f}]",
            f"   有效数据平均值: {np.mean(clean_data):.4f}",
            f"   有效数据标准差: {np.std(clean_data, ddof=1):.4f}"
        ])
        
        return True, original_data, clean_data, blank_count, validation_report, decimal_info

# =============================================
# 统计方法实现
# =============================================

def detect_decimal_places(data):
    """检测数据的小数位数 - 返回最大小数位数"""
    max_decimal_places = 0
    for value in data:
        if isinstance(value, (int, float)) and not np.isnan(value):
            str_value = str(value)
            if 'e' in str_value.lower():
                str_value = format(value, '.15f')
            if '.' in str_value:
                decimal_part = str_value.split('.')[1].rstrip('0')
                current_decimal_places = len(decimal_part)
                max_decimal_places = max(max_decimal_places, current_decimal_places)
    return max_decimal_places

def format_z_scores(z_scores):
    """将Z比分统一格式化为两位小数 - 用于最终展示"""
    if z_scores is None:
        return []
    
    formatted_scores = []
    for score in z_scores:
        try:
            formatted_score = round(float(score), 2)  # 统一格式化为2位小数
            formatted_scores.append(formatted_score)
        except (ValueError, TypeError):
            formatted_scores.append(0.0)
    
    return formatted_scores

def iterative_robust_algorithm(data, max_iterations=50, k=1.5, scheme="strict"):
    """迭代稳健统计法"""
    n = len(data)
    X_star = np.median(data)
    abs_deviations = np.abs(data - X_star)
    median_abs_deviation = np.median(abs_deviations)
    S_star = 1.483 * median_abs_deviation
    
    converged = False
    iteration = 0
    history = []
    
    while iteration < max_iterations and not converged:
        iteration += 1
        prev_X_star = X_star
        prev_S_star = S_star
        
        delta = k * S_star
        Xj_star = np.where(data < X_star - delta, X_star - delta, 
                          np.where(data > X_star + delta, X_star + delta, data))
        
        X_star = np.mean(Xj_star)
        sum_squared_deviations = np.sum((Xj_star - X_star)**2)
        S_star = 1.134 * np.sqrt(sum_squared_deviations / (n-1))
        
        history.append({
            'iteration': iteration,
            'X_star': X_star,
            'S_star': S_star,
            'delta': delta
        })
        
        if (int(prev_X_star * 1000) == int(X_star * 1000) and 
            int(prev_S_star * 1000) == int(S_star * 1000)):
            converged = True
    
    # 根据选择的方案进行格式化
    if scheme == "presentation":
        # 规范展示方案
        decimal_places = detect_decimal_places(data)
        formatted_X_star = round(X_star, decimal_places)
        formatted_S_star = round(S_star, 3)
        
        # 使用格式化后的值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - formatted_X_star) / formatted_S_star
        
        # 使用格式化后的值计算界限
        final_delta = k * formatted_S_star
        lower_limit = formatted_X_star - final_delta
        upper_limit = formatted_X_star + final_delta
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_X_star})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分在结果展示时统一格式化为2位小数。"
        
        robust_mean = formatted_X_star
        robust_std = formatted_S_star
        
    else:
        # 严格计算方案
        decimal_places = 6
        formatted_X_star = X_star
        formatted_S_star = S_star
        
        # 使用原始计算值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - X_star) / S_star
        
        # 计算界限
        final_delta = k * S_star
        lower_limit = X_star - final_delta
        upper_limit = X_star + final_delta
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。Z比分在结果展示时统一格式化为2位小数。"
        
        robust_mean = X_star
        robust_std = S_star
    
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    
    # 安全地分离异常值和正常数据
    outliers_list = []
    clean_data_list = []
    
    for i, value in enumerate(data):
        if outliers_mask[i]:
            outliers_list.append(float(value))
        else:
            clean_data_list.append(float(value))
    
    # 确保Z_scores是安全的Python类型
    safe_z_scores = Z_scores.tolist() if hasattr(Z_scores, 'tolist') else list(Z_scores)
    
    return {
        'robust_mean': float(robust_mean),
        'robust_std': float(robust_std),
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores': safe_z_scores,
        'iterations': iteration,
        'converged': converged,
        'lower_limit': float(lower_limit),
        'upper_limit': float(upper_limit),
        'history': history,
        'method_name': '迭代稳健统计法',
        'formatting_note': formatting_note,
        'calculation_scheme': scheme
    }

def quartile_robust_algorithm(data, scheme="strict"):
    """四分位稳健统计法"""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    niqr = 0.7413 * iqr
    
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    
    outliers_list = []
    clean_data_list = []
    
    for i, value in enumerate(data):
        if outliers_mask[i]:
            outliers_list.append(float(value))
        else:
            clean_data_list.append(float(value))
    
    # 根据选择的方案进行格式化
    if scheme == "presentation":
        # 规范展示方案
        decimal_places = detect_decimal_places(data)
        formatted_median = round(median, decimal_places)
        formatted_niqr = round(niqr, 3)
        
        # 使用格式化后的值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - formatted_median) / formatted_niqr
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_median})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分在结果展示时统一格式化为2位小数。"
        
        robust_mean = formatted_median
        robust_std = formatted_niqr
        
    else:
        # 严格计算方案
        decimal_places = 6
        formatted_median = median
        formatted_niqr = niqr
        
        # 使用原始计算值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - median) / niqr
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。Z比分在结果展示时统一格式化为2位小数。"
        
        robust_mean = median
        robust_std = niqr
    
    # 确保Z_scores是安全的Python类型
    safe_z_scores = Z_scores.tolist() if hasattr(Z_scores, 'tolist') else list(Z_scores)
    
    return {
        'robust_mean': float(robust_mean),
        'robust_std': float(robust_std),
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores': safe_z_scores,
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'niqr': float(niqr),
        'method_name': '四分位稳健统计法',
        'lower_limit': float(lower_limit),
        'upper_limit': float(upper_limit),
        'formatting_note': formatting_note,
        'calculation_scheme': scheme
    }

def q_hampel_robust_algorithm(data, scheme="strict"):
    """Q/Hampel稳健统计方法"""
    n = len(data)
    median = np.median(data)
    
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append(abs(data[i] - data[j]))
    
    if len(pairs) > 0:
        q_std = np.median(pairs) / 1.0484
    else:
        q_std = np.std(data, ddof=1)
    
    current_mean = median
    max_iterations = 10
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        residuals = data - current_mean
        mad = np.median(np.abs(residuals))
        
        if mad == 0:
            break
            
        standardized_residuals = residuals / (1.4826 * mad)
        weights = np.ones_like(data)
        mask1 = np.abs(standardized_residuals) > 1.5
        mask2 = np.abs(standardized_residuals) > 3
        mask3 = np.abs(standardized_residuals) > 4.5
        
        weights[mask1] = 1.5 / np.abs(standardized_residuals[mask1])
        weights[mask2] = 0
        weights[mask3] = 0
        
        new_mean = np.sum(weights * data) / np.sum(weights)
        
        if abs(new_mean - current_mean) < tolerance:
            break
            
        current_mean = new_mean
    
    lower_limit = current_mean - 3 * q_std
    upper_limit = current_mean + 3 * q_std
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    
    outliers_list = []
    clean_data_list = []
    
    for i, value in enumerate(data):
        if outliers_mask[i]:
            outliers_list.append(float(value))
        else:
            clean_data_list.append(float(value))
    
    # 根据选择的方案进行格式化
    if scheme == "presentation":
        # 规范展示方案
        decimal_places = detect_decimal_places(data)
        formatted_current_mean = round(current_mean, decimal_places)
        formatted_q_std = round(q_std, 3)
        
        # 使用格式化后的值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - formatted_current_mean) / formatted_q_std
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_current_mean})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分在结果展示时统一格式化为2位小数。"
        
        robust_mean = formatted_current_mean
        robust_std = formatted_q_std
        
    else:
        # 严格计算方案
        decimal_places = 6
        formatted_current_mean = current_mean
        formatted_q_std = q_std
        
        # 使用原始计算值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - current_mean) / q_std
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。Z比分在结果展示时统一格式化为2位小数。"
        
        robust_mean = current_mean
        robust_std = q_std
    
    # 确保Z_scores是安全的Python类型
    safe_z_scores = Z_scores.tolist() if hasattr(Z_scores, 'tolist') else list(Z_scores)
    
    return {
        'robust_mean': float(robust_mean),
        'robust_std': float(robust_std),
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores': safe_z_scores,
        'method_name': 'Q/Hampel法',
        'lower_limit': float(lower_limit),
        'upper_limit': float(upper_limit),
        'weights': weights.tolist(),
        'formatting_note': formatting_note,
        'calculation_scheme': scheme
    }

# =============================================
# 主程序
# =============================================

# 设置页面
st.set_page_config(
    page_title="统计分析工具",
    page_icon="📊",
    layout="wide"
)

# 初始化会话状态
if 'manual_data' not in st.session_state:
    st.session_state.manual_data = "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'blank_count' not in st.session_state:
    st.session_state.blank_count = 0

# 标题和说明
st.title("📊 统计分析工具")
st.markdown("提供多种稳健统计分析方法，用于处理包含异常值的数据集。")

# 侧边栏设置
st.sidebar.header("⚙️ 分析设置")

method = st.sidebar.selectbox(
    "选择统计方法:",
    ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法"],
    help="选择适合数据特征的稳健统计方法"
)

calculation_scheme = st.sidebar.radio(
    "选择计算方案:",
    ["严格计算方案", "规范展示方案"],
    help="""
    严格计算方案：使用完整精度的计算结果
    规范展示方案：稳健平均值与原始数据小数位数一致，结果更规范
    """
)

if method == "迭代稳健统计法":
    k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
    max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)

# 数据输入
input_method = st.radio("数据输入方式:", ["手动输入", "示例数据"])
data = None

if input_method == "手动输入":
    st.subheader("📝 手动输入数据")
    
    current_data = st.text_area(
        "请输入数据（每行一个数值或用逗号分隔，空白数据会自动忽略）:",
        value=st.session_state.manual_data,
        height=150,
        help="支持空白行、连续逗号或空格表示数据空缺"
    )
    
    col1, col2 = st.columns([1, 1])
    
    def analyze_data():
        """分析数据的回调函数"""
        try:
            is_valid, original_data, clean_data, blank_count, validation_report, decimal_info = \
                DataValidator.comprehensive_validation(
                    current_data, calculation_scheme
                )
            
            if is_valid:
                st.session_state.processed_data = clean_data
                st.session_state.original_data = original_data
                st.session_state.blank_count = blank_count
                st.session_state.data_loaded = True
                st.session_state.validation_report = validation_report
                st.session_state.validation_passed = True
                st.session_state.decimal_info = decimal_info
            else:
                st.session_state.validation_report = validation_report
                st.session_state.validation_passed = False
                st.session_state.data_loaded = False
                
        except Exception as e:
            st.session_state.validation_passed = False
            st.session_state.validation_report = [f"❌ 分析过程中发生错误: {str(e)}"]
            st.session_state.data_loaded = False

    def clear_data():
        """清除数据的回调函数"""
        st.session_state.manual_data = ""
        st.session_state.data_loaded = False
        st.session_state.processed_data = None
        st.session_state.original_data = None
        st.session_state.blank_count = 0

    with col1:
        st.button("分析数据", use_container_width=True, type="primary", on_click=analyze_data)
    with col2:
        st.button("一键清除", use_container_width=True, type="secondary", on_click=clear_data)
    
    # 更新session_state中的数据
    if current_data != st.session_state.manual_data:
        st.session_state.manual_data = current_data
    
    # 数据验证结果显示
    if hasattr(st.session_state, 'validation_report'):
        if st.session_state.validation_passed:
            st.success(f"✅ 数据验证通过！成功解析 {len(st.session_state.processed_data)} 个有效数据点")
        else:
            st.error("❌ 数据验证失败")
        
        with st.expander("📋 查看详细验证报告", expanded=not st.session_state.validation_passed):
            for line in st.session_state.validation_report:
                if line.startswith("❌"):
                    st.error(line)
                elif line.startswith("📊"):
                    st.write("**" + line + "**")
                else:
                    st.write(line)

    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        data = st.session_state.processed_data

else:  # 示例数据
    st.subheader("🎯 示例数据分析")
    example_data = np.array([
        54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
        54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
        54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
        54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
        54.5, 55.9, 53.2, 54.6
    ])
    
    st.write(f"示例数据已加载，包含 {len(example_data)} 个测量值")
    
    with st.expander("📋 查看所有示例数据值", expanded=False):
        df_example = pd.DataFrame({
            '数据编号': range(1, len(example_data) + 1),
            '数值': example_data
        })
        st.dataframe(df_example, use_container_width=True)
    
    data = example_data
    st.session_state.original_data = example_data.tolist()
    st.session_state.blank_count = 0

# 执行分析
if data is not None and len(data) > 0:
    try:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        st.markdown("---")
        st.subheader(f"📈 {method}分析结果")
        
        scheme_display = "规范展示方案" if calculation_scheme == "规范展示方案" else "严格计算方案"
        st.info(f"当前使用: **{scheme_display}**")
        
        # 执行稳健统计分析
        with st.spinner(f"正在执行{method}分析..."):
            scheme_param = "presentation" if calculation_scheme == "规范展示方案" else "strict"
            
            if method == "迭代稳健统计法":
                results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme=scheme_param)
            elif method == "四分位稳健统计法":
                results = quartile_robust_algorithm(data, scheme=scheme_param)
            else:  # Q/Hampel法
                results = q_hampel_robust_algorithm(data, scheme=scheme_param)
            
            # === 关键步骤：统一格式化Z比分为两位小数 ===
            results['Z_scores'] = format_z_scores(results['Z_scores'])
        
        # 显示计算方案说明
        with st.expander("ℹ️ 计算方案说明", expanded=True):
            st.info(results['formatting_note'])
        
        # 显示主要结果
        col1, col2 = st.columns(2)
        with col1:
            st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
            st.metric("稳健标准差", f"{results['robust_std']:.6f}")
        with col2:
            if 'iterations' in results:
                st.metric("迭代次数", results['iterations'])
            st.metric("离群值数量", len(results['outliers']))
        
        # 显示详细结果
        st.subheader("📋 详细结果")
        st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
        
        if len(results['outliers']) > 0:
            st.write(f"**离群值**: {sorted(results['outliers'])}")
        else:
            st.write("**离群值**: 无")
        
        # Z比分数分类统计
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
        
        # 数据可视化
        st.subheader("📊 数据可视化")
        
        # 创建数据框用于可视化
        if input_method == "手动输入" and st.session_state.original_data:
            valid_labels = []
            valid_data = []
            valid_z_scores = []
            
            valid_count = 0
            for i, value in enumerate(st.session_state.original_data):
                if value is not None:
                    label = f"{str(valid_count+1).zfill(3)}"
                    valid_labels.append(label)
                    valid_data.append(float(value))
                    if valid_count < len(results['Z_scores']):
                        valid_z_scores.append(results['Z_scores'][valid_count])
                    valid_count += 1
            
            df_clean = pd.DataFrame({
                'Original_Label': valid_labels,
                'Original_Data': valid_data,
                'Z_Score': valid_z_scores
            })
        else:
            df_clean = pd.DataFrame({
                'Original_Data': data,
                'Z_Score': results['Z_scores']
            })
            valid_labels = [f"{str(i+1).zfill(3)}" for i in range(len(data))]
            df_clean['Original_Label'] = valid_labels
        
        # 创建Z值柱状图
        set_chinese_font()
        
        def classify_data(row):
            try:
                z_score = float(row['Z_Score'])
                if abs(z_score) <= 2:
                    return 'Satisfactory'
                elif 2 < abs(z_score) <= 3:
                    return 'Questionable'
                else:
                    return 'Unsatisfactory'
            except (ValueError, TypeError):
                return 'Unknown'
        
        df_clean['Category'] = df_clean.apply(classify_data, axis=1)
        df_sorted = df_clean.sort_values('Z_Score', ascending=False)
        
        chart_height = max(10, len(df_sorted) * 0.4)
        fig, ax = plt.subplots(figsize=(14, chart_height))
        
        color_map = {
            'Satisfactory': '#00FF00',
            'Questionable': '#FFA500',
            'Unsatisfactory': '#FF0000',
            'Unknown': '#808080'
        }
        
        colors = [color_map.get(cat, '#808080') for cat in df_sorted['Category']]
        
        bars = ax.barh(range(len(df_sorted)), 
                       df_sorted['Z_Score'], 
                       color=colors, 
                       alpha=0.6,
                       height=0.8,
                       edgecolor='white',
                       linewidth=0.5)
        
        # 在柱状图上标注Z值（已经是2位小数）
        for i, (bar, z_value) in enumerate(zip(bars, df_sorted['Z_Score'])):
            try:
                text_color = 'black'
                z_display = f'{float(z_value):.2f}'  # 显示2位小数
                ax.text(bar.get_width() + 0.05 * (1 if bar.get_width() >= 0 else -1), 
                        bar.get_y() + bar.get_height()/2, 
                        z_display, 
                        ha='left' if bar.get_width() >= 0 else 'right', 
                        va='center', fontsize=9, fontweight='bold',
                        color=text_color)
            except:
                continue
        
        ax.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Original Data ID', fontsize=14, fontweight='bold')
        ax.set_title(f'Z-Score Distribution (Sorted)', fontsize=18, fontweight='bold', pad=40)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['Satisfactory'], alpha=0.6, label='Satisfactory (|Z| ≤ 2)'),
            Patch(facecolor=color_map['Questionable'], alpha=0.6, label='Questionable (2 < |Z| ≤ 3)'),
            Patch(facecolor=color_map['Unsatisfactory'], alpha=0.6, label='Unsatisfactory (|Z| > 3)')
        ]
        
        ax.legend(handles=legend_elements, title='Category', title_fontsize=12, fontsize=11, 
                  loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=True)
        
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['Original_Label'])
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        ax.set_facecolor('white')
        
        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # 导出结果
        st.subheader("💾 导出结果")
        
        # 创建结果DataFrame
        result_data = []
        detected_decimal_places = detect_decimal_places(data)
        
        if input_method == "手动输入" and st.session_state.original_data:
            valid_count = 0
            for i, value in enumerate(st.session_state.original_data):
                original_label = f"{str(i+1).zfill(3)}"
                
                if value is not None:
                    z_score = results['Z_scores'][valid_count] if valid_count < len(results['Z_scores']) else None
                    formatted_value = round(float(value), detected_decimal_places) if detected_decimal_places > 0 else int(value)
                    result_data.append({
                        '标签原始标号': original_label,
                        '输入数据': formatted_value,
                        'Z比分数': z_score  # 已经是2位小数
                    })
                    valid_count += 1
                else:
                    result_data.append({
                        '标签原始标号': original_label,
                        '输入数据': None,
                        'Z比分数': None
                    })
        else:
            for i, value in enumerate(data):
                original_label = f"{str(i+1).zfill(3)}"
                z_score = results['Z_scores'][i] if i < len(results['Z_scores']) else None
                formatted_value = round(float(value), detected_decimal_places) if detected_decimal_places > 0 else int(value)
                result_data.append({
                    '标签原始标号': original_label,
                    '输入数据': formatted_value,
                    'Z比分数': z_score  # 已经是2位小数
                })
        
        result_df = pd.DataFrame(result_data)
        
        # 显示预览
        st.write("**导出数据预览:**")
        st.dataframe(result_df, use_container_width=True)
        
        # 创建导出选项
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Excel导出
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                result_df.to_excel(writer, sheet_name='分析数据', index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="📥 下载Excel",
                data=excel_buffer,
                file_name=f"{method}_分析结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with export_col2:
            # CSV导出
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV",
                data=csv_data,
                file_name=f"{method}_分析结果.csv",
                mime="text/csv"
            )
        
        with export_col3:
            # 图表下载
            if 'fig' in locals():
                buffer_png = io.BytesIO()
                fig.savefig(buffer_png, format="png", dpi=300, bbox_inches="tight")
                buffer_png.seek(0)
                st.download_button(
                    label="📥 下载PNG图表",
                    data=buffer_png,
                    file_name=f"z_score_chart_{method}.png",
                    mime="image/png"
                )
        
    except Exception as e:
        st.error(f"❌ 统计分析过程中发生错误: {str(e)}")

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

# 用户反馈
st.markdown("---")
st.subheader("💬 用户反馈")
with st.expander("💬 有问题或建议？点击这里联系我们", expanded=False):
    st.markdown("""
    **技术支持与反馈**
    
    我们重视每一位用户的反馈，如果您遇到以下情况：
    - 使用过程中遇到问题
    - 有功能改进建议
    - 发现数据计算异常
    - 其他任何疑问
    
    请通过以下方式联系我们：
    
    📩 **ypan1104@163.com**
    
    **联系人**：印博士
       
    感谢您帮助我们变得更好！
    """)    