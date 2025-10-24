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
import json
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
# 修改后的数据验证和错误处理模块
# =============================================

class DataValidator:
    """数据验证器类"""
    
    @staticmethod
    def validate_numeric_string_with_blanks(data_string):
        """
        验证数值字符串格式，支持空白数据
        返回: (is_valid, original_data, clean_data, blank_count, error_message)
        """
        if not data_string or data_string.strip() == "":
            return False, [], [], 0, "输入数据不能为空"
        
        # 清理和分割数据
        lines = data_string.strip().split('\n')
        original_data = []  # 包含空白值的原始数据
        clean_data = []     # 清理后的有效数据
        blank_positions = [] # 空白数据的位置
        
        line_num = 0
        for line in lines:
            line_num += 1
            # 支持逗号、空格、分号分隔
            items = re.split(r'[,;\s]+', line.strip())
            col_num = 0
            for item in items:
                col_num += 1
                if item and item.strip():  # 非空字符串
                    try:
                        # 尝试转换为浮点数
                        value = float(item)
                        original_data.append(value)
                        clean_data.append(value)
                    except ValueError:
                        return False, [], [], 0, f"数据格式错误: 第{line_num}行 '{item}' 不是有效的数字"
                else:
                    # 空白数据，记录位置并添加None作为占位符
                    original_data.append(None)
                    blank_positions.append((line_num, col_num))
        
        if len(clean_data) == 0:
            return False, [], [], len(blank_positions), "未找到有效的数值数据"
        
        if len(clean_data) < 3:
            return False, [], [], len(blank_positions), "有效数据点数量不足，至少需要3个有效数据点进行分析"
        
        return True, original_data, np.array(clean_data), len(blank_positions), "数据格式验证通过"
    
    @staticmethod
    def validate_data_range(data_array):
        """
        验证数据范围
        """
        if len(data_array) == 0:
            return False, "没有有效数据可分析"
            
        if np.any(data_array < -1e10):
            return False, f"发现过小数值: {np.min(data_array):.4f}"
        if np.any(data_array > 1e10):
            return False, f"发现过大数值: {np.max(data_array):.4f}"
        
        return True, "数据范围验证通过"
    
    @staticmethod
    def validate_data_variance(data_array):
        """
        验证数据方差，避免所有数据相同
        """
        if len(data_array) == 0:
            return False, "没有有效数据可分析"
            
        if np.std(data_array) == 0:
            return False, "所有有效数据值相同，无法进行统计分析"
        return True, "数据方差验证通过"
    
    @staticmethod
    def detect_potential_outliers(data_array, method='iqr', threshold=3):
        """
        检测潜在异常值
        """
        if len(data_array) < 5:
            return [], "有效数据点不足，无法进行异常值检测"
        
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
        综合数据验证 - 支持空白数据处理
        返回: (is_valid, original_data, clean_data, blank_count, validation_report)
        """
        validation_report = []
        blank_count = 0
        
        # 1. 格式验证（支持空白数据）
        is_valid, original_data, clean_data, blank_count, error_msg = DataValidator.validate_numeric_string_with_blanks(data_string)
        if not is_valid:
            return False, [], [], blank_count, [f"❌ 格式验证失败: {error_msg}"]
        
        validation_report.append(f"✅ {error_msg}")
        
        # 2. 数据范围验证
        is_valid, range_msg = DataValidator.validate_data_range(clean_data)
        if not is_valid:
            return False, original_data, [], blank_count, validation_report + [f"❌ 范围验证失败: {range_msg}"]
        validation_report.append(f"✅ {range_msg}")
        
        # 3. 方差验证
        is_valid, variance_msg = DataValidator.validate_data_variance(clean_data)
        if not is_valid:
            return False, original_data, [], blank_count, validation_report + [f"❌ 方差验证失败: {variance_msg}"]
        validation_report.append(f"✅ {variance_msg}")
        
        # 4. 空白数据统计
        if blank_count > 0:
            validation_report.append(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
        else:
            validation_report.append("✅ 未发现空白数据")
        
        # 5. 异常值检测
        outliers, outliers_info = DataValidator.detect_potential_outliers(clean_data)
        if outliers_info:
            validation_report.append(f"⚠️ {outliers_info[0]}")
            if len(outliers) > 0:
                validation_report.append(f"   异常值: {', '.join([f'{x:.4f}' for x in sorted(outliers)])}")
        else:
            validation_report.append("✅ 未发现明显异常值")
        
        # 6. 数据统计信息（包含空白数据信息）
        validation_report.extend([
            f"📊 数据统计摘要:",
            f"   总数据点数: {len(original_data)}",
            f"   实际可分析数据数: {len(clean_data)}",
            f"   空白数据数: {blank_count}",
            f"   有效数据范围: [{np.min(clean_data):.4f}, {np.max(clean_data):.4f}]",
            f"   有效数据平均值: {np.mean(clean_data):.4f}",
            f"   有效数据标准差: {np.std(clean_data, ddof=1):.4f}"
        ])
        
        return True, original_data, clean_data, blank_count, validation_report

# =============================================
# 修复后的文件格式处理模块
# =============================================

# =============================================
# 修复后的文件格式处理模块
# =============================================

class FileProcessor:
    """文件处理类"""
    
    @staticmethod
    def detect_file_format(uploaded_file):
        """自动检测文件格式"""
        filename = uploaded_file.name.lower()
        
        if filename.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith('.json'):
            return 'json'
        elif filename.endswith('.txt'):
            return 'txt'
        else:
            # 通过内容检测
            content = uploaded_file.read(1024)  # 读取前1024字节
            uploaded_file.seek(0)  # 重置文件指针
            
            try:
                # 尝试解析为JSON
                content_str = content.decode('utf-8')
                json.loads(content_str)
                return 'json'
            except:
                # 尝试解析为CSV
                try:
                    content_str = content.decode('utf-8')
                    # 使用更宽松的CSV解析，允许空白值
                    pd.read_csv(io.StringIO(content_str), na_filter=True, keep_default_na=True)
                    return 'csv'
                except:
                    return 'txt'  # 默认为文本文件
    
    @staticmethod
    def process_excel_file(uploaded_file):
        """处理Excel文件 - 统一使用空白数据处理逻辑"""
        try:
            # 读取Excel文件
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # 如果只有一个工作表，直接读取
            if len(sheet_names) == 1:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0], na_filter=True)
                return df, sheet_names[0], sheet_names
            
            # 多个工作表时让用户选择
            selected_sheet = st.selectbox(
                "选择要分析的工作表:",
                sheet_names,
                help="检测到多个工作表，请选择包含数据的工作表",
                key="excel_sheet_selector"
            )
            
            if selected_sheet:
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, na_filter=True)
                return df, selected_sheet, sheet_names
            else:
                return None, None, sheet_names
                
        except Exception as e:
            st.error(f"Excel文件读取错误: {str(e)}")
            return None, None, []
    
    @staticmethod
    def process_csv_file(uploaded_file):
        """处理CSV文件 - 统一使用空白数据处理逻辑"""
        try:
            # 使用na_filter=True自动识别空白值
            df = pd.read_csv(uploaded_file, na_filter=True)
            return df, "CSV数据", ["CSV数据"]
        except Exception as e:
            st.error(f"CSV文件读取错误: {str(e)}")
            return None, None, []
    
    @staticmethod
    def process_json_file(uploaded_file):
        """处理JSON文件 - 统一使用空白数据处理逻辑"""
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            # 处理不同类型的JSON结构
            if isinstance(data, list):
                # 如果是数组，直接转换为DataFrame
                df = pd.DataFrame(data, columns=['数据'])
                return df, "JSON数组", ["JSON数组"]
            elif isinstance(data, dict):
                # 如果是对象，让用户选择数据字段
                st.info("检测到JSON对象格式，请选择包含数值数据的字段")
                available_keys = list(data.keys())
                selected_key = st.selectbox(
                    "选择数据字段:", 
                    available_keys,
                    key="json_field_selector"
                )
                
                if selected_key and isinstance(data[selected_key], list):
                    df = pd.DataFrame(data[selected_key], columns=[selected_key])
                    return df, f"JSON字段: {selected_key}", available_keys
                else:
                    st.error("选择的字段不包含有效的数值数组")
                    return None, None, available_keys
            else:
                st.error("JSON格式不支持，请提供数组或包含数组的对象")
                return None, None, []
                
        except Exception as e:
            st.error(f"JSON文件解析错误: {str(e)}")
            return None, None, []
    
    @staticmethod
    def process_txt_file(uploaded_file):
        """处理文本文件"""
        try:
            content = uploaded_file.read().decode('utf-8')
            # 将文本内容转换为字符串供验证器使用
            return content, "文本数据", ["文本数据"]
        except Exception as e:
            st.error(f"文本文件读取错误: {str(e)}")
            return None, None, []
    
    @staticmethod
    def extract_data_from_dataframe(df, sheet_name):
        """
        从DataFrame中提取数值数据 - 统一支持空白数据处理
        返回: (clean_data, original_data, blank_count)
        """
        st.info(f"正在从 '{sheet_name}' 中提取数据")
        
        # 显示数据预览
        st.write("**数据预览:**")
        st.dataframe(df.head(), use_container_width=True)
        
        original_data = []  # 包含空白值的原始数据
        clean_data = []     # 清理后的有效数据
        blank_count = 0     # 空白数据计数
        
        # 如果只有一列，直接使用
        if len(df.columns) == 1:
            data_column = df.iloc[:, 0]
            st.write(f"使用唯一列: {df.columns[0]}")
            
            for value in data_column:
                if FileProcessor._is_blank_value(value):  # 修复：使用类名调用静态方法
                    original_data.append(None)
                    blank_count += 1
                else:
                    try:
                        numeric_value = float(value)
                        original_data.append(numeric_value)
                        clean_data.append(numeric_value)
                    except (ValueError, TypeError):
                        original_data.append(None)
                        blank_count += 1
        
        # 多列时让用户选择
        else:
            st.write("检测到多列数据，请选择包含数值数据的列:")
            selected_column = st.selectbox(
                "选择数据列:", 
                df.columns.tolist(),
                key=f"column_selector_{hash(str(df.columns))}"  # 使用唯一的key
            )
            
            if selected_column:
                data_column = df[selected_column]
                
                for value in data_column:
                    if FileProcessor._is_blank_value(value):  # 修复：使用类名调用静态方法
                        original_data.append(None)
                        blank_count += 1
                    else:
                        try:
                            numeric_value = float(value)
                            original_data.append(numeric_value)
                            clean_data.append(numeric_value)
                        except (ValueError, TypeError):
                            original_data.append(None)
                            blank_count += 1
                
                if blank_count > 0:
                    st.warning(f"列 '{selected_column}' 中包含 {blank_count} 个空白或无效数据，已自动过滤")
            else:
                return None, [], 0
        
        return np.array(clean_data), original_data, blank_count
    
    @staticmethod
    def _is_blank_value(value):
        """判断是否为空白值"""
        return (pd.isna(value) or 
                value == "" or 
                value is None or 
                (isinstance(value, str) and value.strip() == ""))
    
    @staticmethod
    def export_to_json(data_array, analysis_results=None, method_name=""):
        """导出数据为JSON格式"""
        export_data = {
            "metadata": {
                "export_time": pd.Timestamp.now().isoformat(),
                "data_points": len(data_array),
                "analysis_method": method_name,
                "software": "稳健统计分析系统"
            },
            "original_data": data_array.tolist()
        }
        
        if analysis_results:
            export_data.update({
                "analysis_results": {
                    "robust_mean": float(analysis_results.get('robust_mean', 0)),
                    "robust_std": float(analysis_results.get('robust_std', 0)),
                    "outliers": [float(x) for x in analysis_results.get('outliers', [])],
                    "z_scores": [float(x) for x in analysis_results.get('Z_scores', [])],
                    "normal_value_range": {
                        "lower_limit": float(analysis_results.get('lower_limit', 0)),
                        "upper_limit": float(analysis_results.get('upper_limit', 0))
                    }
                }
            })
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)

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
        st.session_state.data_history = []
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
        
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
        
    if 'blank_count' not in st.session_state:
        st.session_state.blank_count = 0
        
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    # 创建文本输入框
    current_data = st.text_area(
        "请输入数据（每行一个数值或用逗号分隔，空白数据会自动忽略）:",
        value=st.session_state.manual_data,
        height=150,
        key=f"manual_input_{st.session_state.reset_counter}",
        help="支持空白行、连续逗号或空格表示数据空缺，例如：1, ,2,,3"
    )

    # 更新session_state中的数据
    if current_data != st.session_state.manual_data:
        if st.session_state.manual_data and current_data != st.session_state.manual_data:
            st.session_state.data_history.append(st.session_state.manual_data)
            if len(st.session_state.data_history) > 10:
                st.session_state.data_history = st.session_state.data_history[-10:]
        
        st.session_state.manual_data = current_data

    # 创建操作按钮
    col1, col2, col3 = st.columns([2, 1, 1])

    def clear_data():
        """一键清除数据的回调函数"""
        if st.session_state.manual_data and st.session_state.manual_data.strip():
            st.session_state.data_history.append(st.session_state.manual_data)
        
        st.session_state.manual_data = ""
        st.session_state.data_loaded = False
        st.session_state.processed_data = None
        st.session_state.original_data = None
        st.session_state.blank_count = 0
        st.session_state.reset_counter += 1
        # 清除验证相关状态
        if hasattr(st.session_state, 'validation_report'):
            del st.session_state.validation_report
        if hasattr(st.session_state, 'validation_passed'):
            del st.session_state.validation_passed

    def undo_data():
        """撤销操作的回调函数"""
        if st.session_state.data_history:
            previous_data = st.session_state.data_history.pop()
            st.session_state.manual_data = previous_data
            st.session_state.data_loaded = False
            st.session_state.processed_data = None
            st.session_state.original_data = None
            st.session_state.blank_count = 0
            st.session_state.reset_counter += 1
            # 清除验证相关状态
            if hasattr(st.session_state, 'validation_report'):
                del st.session_state.validation_report
            if hasattr(st.session_state, 'validation_passed'):
                del st.session_state.validation_passed
        else:
            st.session_state.reset_counter += 1

    def analyze_data():
        """分析数据的回调函数 - 支持空白数据处理"""
        try:
            # 使用新的数据验证器（支持空白数据）
            is_valid, original_data, clean_data, blank_count, validation_report = DataValidator.comprehensive_validation(
                st.session_state.manual_data
            )
            
            if is_valid:
                st.session_state.processed_data = clean_data
                st.session_state.original_data = original_data
                st.session_state.blank_count = blank_count
                st.session_state.data_loaded = True
                st.session_state.validation_report = validation_report
                st.session_state.validation_passed = True
            else:
                st.session_state.validation_report = validation_report
                st.session_state.validation_passed = False
                st.session_state.data_loaded = False
                st.session_state.processed_data = None
                st.session_state.original_data = original_data if original_data else []
                st.session_state.blank_count = blank_count
                
        except Exception as e:
            st.session_state.validation_passed = False
            st.session_state.validation_report = [f"❌ 分析过程中发生错误: {str(e)}"]
            st.session_state.data_loaded = False
            st.session_state.processed_data = None
            st.session_state.original_data = None
            st.session_state.blank_count = 0

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
    
    # 数据验证结果显示
    if hasattr(st.session_state, 'validation_report'):
        if st.session_state.validation_passed:
            st.success(f"✅ 数据验证通过！成功解析 {len(st.session_state.processed_data)} 个有效数据点")
            if st.session_state.blank_count > 0:
                st.warning(f"⚠️ 检测到 {st.session_state.blank_count} 个空白数据点，已自动忽略")
        else:
            st.error("❌ 数据验证失败")
        
        with st.expander("📋 查看详细验证报告", expanded=not st.session_state.validation_passed):
            for line in st.session_state.validation_report:
                if line.startswith("❌"):
                    st.error(line)
                elif line.startswith("⚠️"):
                    st.warning(line)
                elif line.startswith("📊"):
                    st.write("**" + line + "**")
                else:
                    st.write(line)
    
    # 调试信息 - 使用安全访问
    with st.expander("调试信息"):
        st.write(f"当前数据: {st.session_state.manual_data}")
        
        # 安全地访问各种状态变量
        original_data_length = len(st.session_state.original_data) if st.session_state.original_data is not None else 0
        processed_data_length = len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0
        blank_count = st.session_state.blank_count
        history_length = len(st.session_state.data_history)
        reset_counter = st.session_state.reset_counter
        
        st.write(f"原始数据长度: {original_data_length}")
        st.write(f"有效数据长度: {processed_data_length}")
        st.write(f"空白数据数: {blank_count}")
        st.write(f"历史记录长度: {history_length}")
        st.write(f"重置计数器: {reset_counter}")

    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        data = st.session_state.processed_data

elif input_method == "文件上传":
    st.subheader("📁 上传数据文件")
    
    # 扩展支持的文件类型
    uploaded_file = st.file_uploader(
        "选择数据文件 (支持 CSV、TXT、Excel、JSON)", 
        type=['csv', 'txt', 'xlsx', 'xls', 'json'],
        help="支持多种文件格式：CSV、文本文件、Excel工作簿、JSON数据文件。空白数据会自动识别并忽略。"
    )
    
    # 文件格式说明
    with st.expander("📝 查看文件格式说明和示例", expanded=False):
        st.markdown("""
        **支持的文件格式:**
        
        **📄 TXT文件**
        - 每行一个数值
        - 支持整数和小数
        - 空白行、连续逗号或空格表示数据空缺
        - 示例：`1, ,2,,3` 表示有2个空白数据
        
        **📊 CSV文件**  
        - 第一列包含数值数据
        - 可以有表头，也可以没有
        - 空白单元格表示数据空缺
        
        **📑 Excel文件 (xlsx, xls)**
        - 支持多工作表
        - 自动检测数据列
        - 支持数值数据提取
        - 空白单元格表示数据空缺
        
        **📋 JSON文件**
        - 支持数组格式: `[1.1, 2.2, 3.3]`
        - 支持对象格式: `{"data": [1.1, 2.2, 3.3]}`
        - 自动识别数据结构
        - `null` 值表示数据空缺

        **空白数据处理：**
        - 空白数据会保留原始标签
        - 分析时自动忽略空白数据
        - 导出时显示空白数据统计
        """)
        
        # 提供多种示例文件下载
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            example_content = "54.4\n \n54.6\n54.2\n\n54.3\n53.9"
            st.download_button(
                label="下载TXT示例",
                data=example_content,
                file_name="example_data_with_blanks.txt",
                mime="text/plain",
                help="包含空白数据的TXT示例文件"
            )
        
        with col2:
            # 创建CSV示例（包含空白数据）
            csv_data = "测量值\n54.4\n \n54.6\n54.2\n\n54.3\n53.9"
            st.download_button(
                label="下载CSV示例",
                data=csv_data,
                file_name="example_data_with_blanks.csv",
                mime="text/csv",
                help="包含空白数据的CSV示例文件"
            )
        
        with col3:
            # 创建JSON示例（包含空白数据）
            json_data = [54.4, None, 54.6, 54.2, None, 54.3, 53.9]
            st.download_button(
                label="下载JSON示例",
                data=json.dumps(json_data, indent=2),
                file_name="example_data_with_blanks.json",
                mime="application/json",
                help="包含空白数据的JSON示例文件"
            )
        
        with col4:
            # 创建Excel示例（包含空白数据）
            df_example = pd.DataFrame({'测量值': [54.4, None, 54.6, 54.2, None, 54.3, 53.9]})
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_example.to_excel(writer, index=False, sheet_name='测量数据')
            excel_buffer.seek(0)
            
            st.download_button(
                label="下载Excel示例",
                data=excel_buffer,
                file_name="example_data_with_blanks.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="包含空白数据的Excel示例文件"
            )
    
    if uploaded_file is not None:
        try:
            # 显示文件信息
            file_size = uploaded_file.size / 1024  # KB
            st.write(f"📄 **文件信息**: {uploaded_file.name} ({file_size:.1f} KB)")
            
            # 自动检测文件格式
            file_format = FileProcessor.detect_file_format(uploaded_file)
            st.write(f"🔍 **检测到的格式**: {file_format.upper()}")
            
            # 初始化会话状态
            if 'file_processed_data' not in st.session_state:
                st.session_state.file_processed_data = None
            if 'file_original_data' not in st.session_state:
                st.session_state.file_original_data = None
            if 'file_blank_count' not in st.session_state:
                st.session_state.file_blank_count = 0
            if 'file_validation_report' not in st.session_state:
                st.session_state.file_validation_report = []
            if 'file_validation_passed' not in st.session_state:
                st.session_state.file_validation_passed = False
            
            processed_data = None
            original_data = []
            blank_count = 0
            validation_content = ""
            
            # 处理任何文件格式
            df, sheet_name, all_sheets = FileProcessor.process_excel_file(uploaded_file)  # 或 process_csv_file, process_json_file
            
            if df is not None:
                # 提取数据（自动支持空白数据处理）
                clean_data, original_data, blank_count = FileProcessor.extract_data_from_dataframe(df, sheet_name)
                
                if clean_data is not None:
                    # 使用数据验证器验证
                    is_valid, validated_original_data, validated_clean_data, validated_blank_count, validation_report = DataValidator.comprehensive_validation(
                        "\n".join([str(x) if x is not None else "" for x in original_data])
                    )            
            
            # 数据验证和结果展示
            if processed_data is not None and len(processed_data) > 0:
                st.session_state.file_processed_data = processed_data
                
                st.success(f"✅ 文件验证通过！成功加载 {len(processed_data)} 个有效数据点")
                if blank_count > 0:
                    st.warning(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
                
                # 显示验证报告
                with st.expander("📋 查看文件验证报告", expanded=True):
                    for line in st.session_state.file_validation_report:
                        if line.startswith("❌"):
                            st.error(line)
                        elif line.startswith("⚠️"):
                            st.warning(line)
                        elif line.startswith("📊"):
                            st.write("**" + line + "**")
                        else:
                            st.write(line)
                
                st.write("**前10个有效数据:**", processed_data[:10])
                
                # 设置数据变量，以便后续分析
                data = processed_data
                st.session_state.original_data = original_data
                st.session_state.blank_count = blank_count
                
            else:
                st.error("❌ 文件数据验证失败或没有有效数据")
                if hasattr(st.session_state, 'file_validation_report'):
                    with st.expander("📋 查看验证详情", expanded=True):
                        for line in st.session_state.file_validation_report:
                            if line.startswith("❌"):
                                st.error(line)
                            else:
                                st.write(line)
            
        except Exception as e:
            st.error(f"❌ 文件处理错误: {str(e)}")
            st.info("💡 请确保文件格式正确且包含有效的数值数据")

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
    
    st.write(f"示例数据已加载，包含 {len(example_data)} 个测量值")
    
    if is_valid:
        st.success("✅ 示例数据验证通过")
    
    with st.expander("📋 查看所有示例数据值", expanded=False):
        df_example = pd.DataFrame({
            '数据编号': range(1, len(example_data) + 1),
            '数值': example_data
        })
        st.dataframe(df_example, use_container_width=True)
        
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
    
    data = example_data

# 方法描述（保持不变）
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

# 统计方法实现（保持不变）
def iterative_robust_algorithm(data, max_iterations=50, k=1.5):
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
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    Z_scores = (data - median) / niqr
    
    return {
        'robust_mean': median,
        'robust_std': niqr,
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
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    Z_scores = (data - current_mean) / q_std
    
    return {
        'robust_mean': current_mean,
        'robust_std': q_std,
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
        
        # 数据分布可视化
        st.subheader("输入数据正态分布分析")
        
        dist_col1, dist_col2 = st.columns([1, 2])
        
        with dist_col1:
            st.write("**数据统计摘要:**")
            st.write(f"数据点数: {len(data)}")
            st.write(f"平均值: {np.mean(data):.4f}")
            st.write(f"标准差: {np.std(data, ddof=1):.4f}")
            st.write(f"最小值: {np.min(data):.4f}")
            st.write(f"最大值: {np.max(data):.4f}")
            st.write(f"中位数: {np.median(data):.4f}")
            
            from scipy.stats import shapiro
            if len(data) >= 3 and len(data) <= 5000:
                stat, p_value = shapiro(data)
                st.write(f"正态性检验p值: {p_value:.4f}")
                if p_value > 0.05:
                    st.write(":green[数据符合正态分布 (p > 0.05)]")
                else:
                    st.write(":red[数据可能不符合正态分布 (p ≤ 0.05)]")
        
        with dist_col2:
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax_dist.hist(data, bins=15, alpha=0.7, color='skyblue', 
                                           edgecolor='black', density=True, label='Data Distribution')
            
            from scipy.stats import norm
            xmin, xmax = ax_dist.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, np.mean(data), np.std(data, ddof=1))
            ax_dist.plot(x, p, 'k', linewidth=2, label='Normal Distribution Curve')
            
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
                else:
                    results = q_hampel_robust_algorithm(data)
                
                # 结果显示（保持不变）
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
                    st.metric("稳健标准差", f"{results['robust_std']:.6f}")
                with col2:
                    if 'iterations' in results:
                        st.metric("迭代次数", results['iterations'])
                    st.metric("离群值数量", len(results['outliers']))
                
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
                
                st.subheader("📋 详细结果")
                st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
                if 'converged' in results:
                    st.write(f"**收敛状态**: {'是' if results['converged'] else '否'}")
                
                if len(results['outliers']) > 0:
                    outliers_list = [float(x) for x in sorted(results['outliers'])]
                    st.write(f"**离群值**: {outliers_list}")
                else:
                    st.write("**离群值**: 无")
                
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
                
                # 生成三位数字标签 - 仅对有效数据
                valid_labels = []
                valid_data_count = 0
                
                # 遍历原始数据，只为有效数据生成标签
                for i, value in enumerate(st.session_state.original_data):
                    if value is not None:  # 有效数据
                        label = f"{str(valid_data_count+1).zfill(3)}"  # 001, 002, ...
                        valid_labels.append(label)
                        valid_data_count += 1
                
                # 将标签添加到数据框
                df_clean['Original_Label'] = valid_labels
                
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
                ax.set_ylabel('原始标签', fontsize=14, fontweight='bold')  # 修改y轴标签
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
                
                # 设置Y轴刻度 - 使用三位数字标签
                ax.set_yticks(y_positions)
                ax.set_yticklabels(df_sorted['Original_Label'])  # 使用三位数字标签
                
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

                # =============================================
                # 修改后的导出结果模块 - 使用三位数字标签
                # =============================================
                
                # 导出功能
                st.subheader("💾 导出结果")
                
                # 创建结果DataFrame - 使用三位数字标签
                result_data = []
                valid_data_count = 0
                
                # 处理原始数据（包括空白值）
                for i, value in enumerate(st.session_state.original_data):
                    # 修改：使用三位数字标签，如001, 002, ..., 010, 011, ...
                    original_label = f"{str(i+1).zfill(3)}"  # 001, 002, ..., 099, 100, ...
                    
                    if value is not None:  # 有效数据
                        z_score = results['Z_scores'][valid_data_count] if valid_data_count < len(results['Z_scores']) else None
                        result_data.append({
                            '标签原始标号': original_label,  # 修改列名以匹配新格式
                            '输入数据': round(value, 2),  # 保留两位小数
                            'Z比分数': round(z_score, 2) if z_score is not None else None  # 保留两位小数
                        })
                        valid_data_count += 1
                    else:  # 空白数据
                        result_data.append({
                            '标签原始标号': original_label,  # 修改列名以匹配新格式
                            '输入数据': None,  # 空白数据
                            'Z比分数': None   # 空白数据没有Z分数
                        })
                
                result_df = pd.DataFrame(result_data)
                
                # 计算统计量 - 更新为新的命名
                total_data_count = len(st.session_state.original_data)  # 总数据数
                blank_data_count = st.session_state.blank_count  # 空白数据数
                actual_analyzable_count = len(data)  # 实际可分析数据数
                
                stats_data = {
                    '统计量名称': ['总数据数', '实际可分析数据数', '空白数据数', '指定值', '能力评定标准差', '最小值', '最大值', '极差'],
                    '数值': [
                        total_data_count,  # 总数据数
                        actual_analyzable_count,  # 实际可分析数据数
                        blank_data_count,  # 空白数据数
                        round(results['robust_mean'], 2),  # 指定值（稳健平均值）
                        round(results['robust_std'], 2),  # 能力评定标准差（稳健标准差）
                        round(np.min(data), 2),  # 最小值
                        round(np.max(data), 2),  # 最大值
                        round(np.max(data) - np.min(data), 2)  # 极差
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                
                # 显示预览
                st.write("**导出数据预览:**")
                st.dataframe(result_df, use_container_width=True)
                
                st.write("**统计量摘要:**")
                st.dataframe(stats_df, use_container_width=True)
                
                # 创建多格式导出选项
                export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                
                with export_col1:
                    # CSV导出
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载CSV",
                        data=csv_data,
                        file_name=f"{method}_分析结果.csv",
                        mime="text/csv",
                        help="下载CSV格式的分析结果表格（使用三位数字标签）"
                    )
                
                with export_col2:
                    # JSON导出 - 修改为新的数据结构
                    export_data = {
                        "metadata": {
                            "export_time": pd.Timestamp.now().isoformat(),
                            "analysis_method": method,
                            "software": "稳健统计分析系统",
                            "data_summary": {
                                "total_data_points": total_data_count,
                                "actual_analyzable_data": actual_analyzable_count,
                                "blank_data_points": blank_data_count
                            }
                        },
                        "data_table": result_df.to_dict('records'),
                        "statistics": stats_df.set_index('统计量名称')['数值'].to_dict()
                    }
                    
                    json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 下载JSON",
                        data=json_data,
                        file_name=f"{method}_分析结果.json",
                        mime="application/json",
                        help="下载JSON格式的分析结果和数据（使用三位数字标签）"
                    )
                
                with export_col3:
                    # Excel导出 - 修改为新的工作表结构
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # 数据表格工作表
                        result_df.to_excel(writer, sheet_name='分析数据', index=False)
                        
                        # 统计量工作表
                        stats_df.to_excel(writer, sheet_name='统计摘要', index=False)
                        
                        # 详细信息工作表
                        detail_data = {
                            '项目': ['分析方法', '总数据点数', '实际可分析数据数', '空白数据数', 
                                   '稳健平均值', '稳健标准差', '离群值数量', '正常值下限', '正常值上限'],
                            '数值': [method, total_data_count, actual_analyzable_count, blank_data_count,
                                   results['robust_mean'], results['robust_std'], len(results['outliers']), 
                                   results['lower_limit'], results['upper_limit']]
                        }
                        pd.DataFrame(detail_data).to_excel(writer, sheet_name='详细信息', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 下载Excel",
                        data=excel_buffer,
                        file_name=f"{method}_分析结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="下载Excel工作簿，包含分析数据和统计摘要（使用三位数字标签）"
                    )
                
                with export_col4:
                    # 文本报告导出 - 修改为新的格式
                    report = f"""
{method}分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

数据概览:
--------
总数据点数: {total_data_count}
实际可分析数据数: {actual_analyzable_count}
空白数据数: {blank_data_count}

数据表格:
--------
标签原始标号	输入数据	Z比分数
"""
                    # 添加数据行
                    for i in range(len(result_df)):
                        row = result_df.iloc[i]
                        input_data = "" if pd.isna(row['输入数据']) else f"{row['输入数据']}"
                        z_score = "" if pd.isna(row['Z比分数']) else f"{row['Z比分数']}"
                        report += f"{row['标签原始标号']}\t{input_data}\t{z_score}\n"
                    
                    report += f"""
统计量摘要:
----------
"""
                    for stat_name, value in stats_df.set_index('统计量名称')['数值'].items():
                        report += f"{stat_name}: {value}\n"
                    
                    report += f"""
分析详情:
--------
分析方法: {method}
离群值数量: {len(results['outliers'])}
正常值范围: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]
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
                    
                    # Z比分数分类统计
                    z_scores_abs = np.abs(results['Z_scores'])
                    satisfactory = np.sum(z_scores_abs <= 2)
                    questionable = np.sum((z_scores_abs > 2) & (z_scores_abs <= 3))
                    unsatisfactory = np.sum(z_scores_abs > 3)
                    
                    report += f"""
Z比分数分类（仅有效数据）:
-------------------------
满意 (|Z| ≤ 2): {satisfactory} 个数据点
可疑 (2 < |Z| ≤ 3): {questionable} 个数据点  
不满意 (|Z| > 3): {unsatisfactory} 个数据点

离群值列表:
----------
"""
                    if len(results['outliers']) > 0:
                        outliers_list = [f"{float(x):.2f}" for x in sorted(results['outliers'])]
                        report += f"{', '.join(outliers_list)}"
                    else:
                        report += "无"
                    
                    st.download_button(
                        label="📥 下载报告",
                        data=report,
                        file_name=f"{method}_分析报告.txt",
                        mime="text/plain",
                        help="下载文本格式的详细分析报告（使用三位数字标签）"
                    )
                
                # 图表下载功能保持不变
                st.subheader("📊 下载图表")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    buffer_png = io.BytesIO()
                    fig.savefig(buffer_png, format="png", dpi=300, bbox_inches="tight")
                    buffer_png.seek(0)
                    st.download_button(
                        label="📥 下载PNG图表",
                        data=buffer_png,
                        file_name=f"z_score_chart_{method}.png",
                        mime="image/png"
                    )
                
                with chart_col2:
                    buffer_pdf = io.BytesIO()
                    fig.savefig(buffer_pdf, format="pdf", bbox_inches="tight")
                    buffer_pdf.seek(0)
                    st.download_button(
                        label="📥 下载PDF图表",
                        data=buffer_pdf,
                        file_name=f"z_score_chart_{method}.pdf",
                        mime="application/pdf"
                    )
                
                st.info("💡 提示：导出的数据表格使用三位数字标签格式（001、002...），空白数据会保留标签但数据为空") 
                
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
    
    📩 **xxx@163.com**
    
    **联系人**：x博士
       
    感谢您帮助我们变得更好！
    """)