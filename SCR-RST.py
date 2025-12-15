import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
import re
import base64
from collections import Counter
from PIL import Image
from scipy.stats import norm
from scipy import interpolate

# =============================================
# 初始化会话状态
# =============================================

def initialize_session_state():
    """初始化所有必要的会话状态变量"""
    defaults = {
        'manual_data': "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6",
        'data_history': [],
        'data_loaded': False,
        'processed_data': None,
        'original_data': None,
        'blank_count': 0,
        'reset_counter': 0,
        'validation_report': [],
        'validation_passed': False,
        'decimal_info': {},
        'two_column_data': "",
        'two_column_processed': False,
        'label_data_pairs': [],
        'two_column_validation_report': [],
        'valid_pairs': [],
        'original_labels': [],
        'file_processed_data': None,
        'file_original_data': None,
        'file_blank_count': 0,
        'file_validation_report': [],
        'file_validation_passed': False,
        'file_decimal_info': {},
        'two_column_decimal_info': {},
        'calculation_scheme': "严格计算方案",
        'robust_mean_input': "54.4",
        'robust_std_input': "0.3"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================
# 配置函数
# =============================================

def set_chinese_font():
    """设置中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.warning(f"字体设置失败: {e}")

# =============================================
# Z比分工具函数
# =============================================

def classify_z_score(z_score):
    """根据Z比分进行分类"""
    try:
        z_abs = abs(float(z_score))
        if z_abs <= 2:
            return "满意"
        elif 2 < z_abs < 3:
            return "可疑"
        else:
            return "不满意"
    except (ValueError, TypeError):
        return "未知"

def format_z_score_display(z_score):
    """将单个Z比分格式化为两位小数显示"""
    if z_score is None or pd.isna(z_score):
        return ""
    try:
        return f"{float(z_score):.2f}"
    except (ValueError, TypeError):
        return "0.00"

def format_z_scores(z_scores):
    """将Z比分列表统一格式化为两位小数"""
    if z_scores is None:
        return None
    
    formatted_scores = []
    for score in z_scores:
        try:
            formatted_score = format_z_score_display(score)
            formatted_scores.append(formatted_score)
        except (ValueError, TypeError):
            formatted_scores.append("0.00")
    
    return formatted_scores

# =============================================
# 数据验证器
# =============================================

class DataValidator:
    """统一数据验证器"""
    
    @staticmethod
    def validate_numeric_string_with_blanks(data_string):
        """
        验证数值字符串格式，支持空白数据
        返回: (is_valid, original_data, clean_data, blank_count, error_message, decimal_info)
        """
        if data_string is None:
            return False, [], [], 0, "输入数据为None", {}
        
        if not isinstance(data_string, str):
            return False, [], [], 0, f"输入数据类型错误: {type(data_string)}，应为字符串", {}
        
        if not data_string or data_string.strip() == "":
            return False, [], [], 0, "输入数据不能为空", {}
        
        lines = data_string.strip().split('\n')
        original_data = []
        clean_data = []
        blank_count = 0
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': 0
        }
        
        max_decimal_places = 0
        previous_decimal_places = None
        
        for line_num, line in enumerate(lines, 1):
            items = re.split(r'[,;\s]+', line.strip())
            for col_num, item in enumerate(items, 1):
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
                        
                        if previous_decimal_places is not None and previous_decimal_places != decimal_places:
                            decimal_info['consistent_decimals'] = False
                        previous_decimal_places = decimal_places
                        
                    except ValueError:
                        return False, [], [], 0, f"数据格式错误: 第{line_num}行 '{item}' 不是有效的数字", {}
                else:
                    original_data.append(None)
                    blank_count += 1
        
        decimal_info['detected_decimal_places'] = max_decimal_places
        return True, original_data, clean_data, blank_count, "数据格式验证通过", decimal_info
    
    @staticmethod
    def validate_data_range(data_array):
        """验证数据范围"""
        if len(data_array) == 0:
            return False, "数据数组为空"
        
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        if min_val == max_val:
            return False, "所有数据值相同，无法进行统计分析"
        
        return True, f"数据范围验证通过: [{min_val:.4f}, {max_val:.4f}]"
    
    @staticmethod
    def validate_data_variance(data_array):
        """验证数据方差"""
        if len(data_array) < 2:
            return False, "数据点不足，无法计算方差"
        
        variance = np.var(data_array, ddof=1)
        if variance == 0:
            return False, "数据方差为零，所有数据点相同"
        
        return True, f"数据方差验证通过: {variance:.6f}"
    
    @staticmethod
    def detect_potential_outliers(data_array):
        """检测潜在异常值"""
        try:
            if len(data_array) < 3:
                return [], ["数据点不足，无法进行异常值检测"]
            
            if not isinstance(data_array, np.ndarray):
                data_array = np.array(data_array)
            
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_list = []
            for value in data_array:
                if value < lower_bound or value > upper_bound:
                    outliers_list.append(float(value))
            
            if len(outliers_list) > 0:
                return outliers_list, [f"检测到 {len(outliers_list)} 个潜在异常值（基于IQR方法）"]
            else:
                return [], ["未发现明显异常值"]
                
        except Exception as e:
            return [], [f"异常值检测过程中发生错误: {str(e)}"]
    
    @staticmethod
    def validate_calculation_scheme_compatibility(data_array, calculation_scheme, decimal_info):
        """验证计算方案与数据的兼容性"""
        validation_messages = []
        
        if calculation_scheme == "规范展示方案":
            if not decimal_info.get('consistent_decimals', True):
                validation_messages.append(
                    "⚠️ 检测到数据的小数位数不一致，规范展示方案将使用最常出现的小数位数进行格式化"
                )
            
            if decimal_info.get('max_decimal_places', 0) > 4:
                validation_messages.append(
                    f"⚠️ 数据包含最多 {decimal_info['max_decimal_places']} 位小数，规范展示方案可能损失部分精度"
                )
        
        elif calculation_scheme == "严格计算方案":
            validation_messages.append(
                "✅ 严格计算方案将保留完整计算精度"
            )
        
        return validation_messages
    
    @staticmethod
    def get_recommended_scheme(decimal_info):
        """根据数据特征推荐计算方案"""
        if decimal_info.get('consistent_decimals', True) and decimal_info.get('max_decimal_places', 0) <= 2:
            return "规范展示方案", "数据小数位数一致且较少，推荐使用规范展示方案"
        else:
            return "严格计算方案", "数据小数位数不一致或较多，推荐使用严格计算方案确保精度"
    
    @staticmethod
    def comprehensive_validation(data_string, calculation_scheme="规范展示方案"):
        """
        综合数据验证 - 支持空白数据处理和计算方案验证
        返回: (is_valid, original_data, clean_data, blank_count, validation_report, decimal_info)
        """
        validation_report = []
        
        # 1. 格式验证（支持空白数据）
        is_valid, original_data, clean_data, blank_count, error_msg, decimal_info = \
            DataValidator.validate_numeric_string_with_blanks(data_string)
        
        if not is_valid:
            return False, [], [], blank_count, [f"❌ 格式验证失败: {error_msg}"], decimal_info
        
        validation_report.append(f"✅ {error_msg}")
        
        # 2. 数据范围验证
        is_valid, range_msg = DataValidator.validate_data_range(clean_data)
        if not is_valid:
            return False, original_data, [], blank_count, validation_report + [f"❌ 范围验证失败: {range_msg}"], decimal_info
        validation_report.append(f"✅ {range_msg}")
        
        # 3. 方差验证
        is_valid, variance_msg = DataValidator.validate_data_variance(clean_data)
        if not is_valid:
            return False, original_data, [], blank_count, validation_report + [f"❌ 方差验证失败: {variance_msg}"], decimal_info
        validation_report.append(f"✅ {variance_msg}")
        
        # 4. 计算方案兼容性验证
        scheme_messages = DataValidator.validate_calculation_scheme_compatibility(
            clean_data, calculation_scheme, decimal_info
        )
        validation_report.extend(scheme_messages)
        
        # 5. 推荐计算方案
        recommended_scheme, recommendation_reason = DataValidator.get_recommended_scheme(decimal_info)
        validation_report.append(f"💡 推荐计算方案: {recommended_scheme} - {recommendation_reason}")
        
        # 6. 空白数据统计
        if blank_count > 0:
            validation_report.append(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
        else:
            validation_report.append("✅ 未发现空白数据")
        
        # 7. 小数位数统计
        if decimal_info['decimal_places_count']:
            decimal_stats = ", ".join([f"{places}位({count}个)" for places, count in decimal_info['decimal_places_count'].items()])
            validation_report.append(f"📊 数据小数位数分布: {decimal_stats}")
            validation_report.append(f"📏 使用的小数位数: {decimal_info['detected_decimal_places']}位（基于最大小数位数）")
            validation_report.append(f"🔍 小数位数一致性: {'是' if decimal_info['consistent_decimals'] else '否'}")
            
            if not decimal_info['consistent_decimals']:
                validation_report.append("⚠️ 检测到数据中小数位数不一致，将使用最大小数位数作为输出格式标准")
        else:
            validation_report.append("📊 数据小数位数: 均为整数")
            validation_report.append("📏 使用的小数位数: 0位（整数格式）")
        
        # 8. 异常值检测
        outliers, outliers_info = DataValidator.detect_potential_outliers(clean_data)
        if outliers_info and "检测到" in outliers_info[0]:
            validation_report.append(f"⚠️ {outliers_info[0]}")
            if outliers:
                validation_report.append(f"   异常值: {', '.join([f'{x:.4f}' for x in outliers])}")
        else:
            validation_report.append("✅ 未发现明显异常值")
        
        # 9. 数据统计信息
        validation_report.extend([
            f"📈 数据统计摘要:",
            f"   总数据点数: {len(original_data)}",
            f"   实际可分析数据数: {len(clean_data)}",
            f"   空白数据数: {blank_count}",
            f"   有效数据范围: [{np.min(clean_data):.4f}, {np.max(clean_data):.4f}]",
            f"   有效数据平均值: {np.mean(clean_data):.4f}",
            f"   有效数据标准差: {np.std(clean_data, ddof=1):.4f}"
        ])
        
        return True, original_data, clean_data, blank_count, validation_report, decimal_info

# =============================================
# 两列数据验证
# =============================================

def validate_two_column_data(two_column_input, calculation_scheme):
    """验证两列数据输入"""
    try:
        lines = two_column_input.strip().split('\n')
        label_data_pairs = []
        valid_pairs = []
        invalid_lines = []
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': 0
        }
        max_decimal_places = 0
        previous_decimal_places = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                parts = [part.strip() for part in re.split(r'[,;\s]+', line) if part.strip()]
                if len(parts) >= 2:
                    label = parts[0]
                    value_str = parts[-1]
                    try:
                        value = float(value_str)
                        label_data_pairs.append((label, value))
                        valid_pairs.append((label, value))
                        
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
                        
                        if previous_decimal_places is not None and previous_decimal_places != decimal_places:
                            decimal_info['consistent_decimals'] = False
                        previous_decimal_places = decimal_places
                        
                    except ValueError:
                        invalid_lines.append(f"第{i+1}行: '{value_str}' 不是有效的数字")
                else:
                    invalid_lines.append(f"第{i+1}行: 格式错误，应为'标签 数值'或'标签,数值'，当前内容: '{line}'")
        
        decimal_info['detected_decimal_places'] = max_decimal_places
        
        return label_data_pairs, valid_pairs, invalid_lines, decimal_info
        
    except Exception as e:
        return [], [], [f"数据处理错误: {str(e)}"], {}

# =============================================
# 文件处理器
# =============================================

class FileProcessor:
    """文件处理类"""
    
    @staticmethod
    def _is_blank_value(value):
        """检查值是否为空白值"""
        if value is None:
            return True
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False
    
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
            return 'txt'
    
    @staticmethod
    def process_excel_file(uploaded_file):
        """处理Excel文件"""
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0], na_filter=True)
                return df, sheet_names[0], sheet_names
            else:
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
        """处理CSV文件"""
        try:
            df = pd.read_csv(uploaded_file, na_filter=True)
            return df, "CSV数据", ["CSV数据"]
        except Exception as e:
            st.error(f"CSV文件读取错误: {str(e)}")
            return None, None, []
    
    @staticmethod
    def process_json_file(uploaded_file):
        """处理JSON文件"""
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                return df, "JSON数组", ["JSON数组"]
            elif isinstance(data, dict):
                st.info("检测到JSON对象格式，请选择包含数值数据的字段")
                available_keys = list(data.keys())
                selected_key = st.selectbox(
                    "选择数据字段:", 
                    available_keys,
                    key="json_field_selector"
                )
                
                if selected_key:
                    if isinstance(data[selected_key], list):
                        df = pd.DataFrame(data[selected_key])
                        return df, f"JSON字段: {selected_key}", available_keys
                    elif isinstance(data[selected_key], dict):
                        nested_keys = list(data[selected_key].keys())
                        selected_nested_key = st.selectbox(
                            "选择嵌套数据字段:", 
                            nested_keys,
                            key="json_nested_field_selector"
                        )
                        if isinstance(data[selected_key][selected_nested_key], list):
                            df = pd.DataFrame(data[selected_key][selected_nested_key])
                            return df, f"JSON字段: {selected_key}.{selected_nested_key}", available_keys
                        else:
                            st.error("选择的嵌套字段不包含有效的数值数组")
                            return None, None, available_keys
                    else:
                        st.error("选择的字段不包含有效的数值数组")
                        return None, None, available_keys
                else:
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
            separators = [',', '\t', ';', '|', ' ']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep, na_filter=True, engine='python')
                    if len(df.columns) > 1:
                        st.info(f"检测到文本文件使用分隔符: '{sep}'")
                        break
                except:
                    continue
            
            if df is None:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=',', na_filter=True)
                except:
                    lines = content.strip().split('\n')
                    data = []
                    for line in lines:
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                        if numbers:
                            data.extend([float(num) for num in numbers])
                    df = pd.DataFrame(data, columns=['数据'])
            
            return df, "文本数据", ["文本数据"]
        except Exception as e:
            st.error(f"文本文件读取错误: {str(e)}")
            return None, None, []
    
    @staticmethod
    def extract_data_from_dataframe(df, sheet_name):
        """从DataFrame中提取数值数据"""
        st.info(f"正在从 '{sheet_name}' 中提取数据")
        st.write("**数据预览:**")
        st.dataframe(df.head(), use_container_width=True)
        
        original_data = []
        clean_data = []
        blank_count = 0
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': 0
        }
        max_decimal_places = 0
        
        if len(df.columns) == 1:
            data_column = df.iloc[:, 0]
            st.write(f"使用唯一列: {df.columns[0]}")
        else:
            col_info = []
            for col in df.columns:
                sample_values = df[col].dropna().head(3).tolist()
                dtype = df[col].dtype
                col_info.append(f"{col} (类型: {dtype}, 样例: {sample_values})")
            
            selected_column = st.selectbox(
                "选择数据列:", 
                df.columns.tolist(),
                format_func=lambda x: f"{x} (类型: {df[x].dtype}, 样例: {df[x].dropna().head(3).tolist()})",
                key=f"column_selector_{hash(str(df.columns))}"
            )
            
            if selected_column:
                data_column = df[selected_column]
                st.success(f"已选择列: {selected_column}")
            else:
                st.error("请选择数据列")
                return None, [], 0, decimal_info
        
        for value in data_column:
            if FileProcessor._is_blank_value(value):
                original_data.append(None)
                blank_count += 1
            else:
                try:
                    numeric_value = float(value)
                    original_data.append(numeric_value)
                    clean_data.append(numeric_value)
                    
                    str_value = str(numeric_value)
                    if '.' in str_value:
                        decimal_part = str_value.split('.')[1].rstrip('0')
                        decimal_places = len(decimal_part)
                    else:
                        decimal_places = 0
                    
                    max_decimal_places = max(max_decimal_places, decimal_places)
                    decimal_info['decimal_places_count'][decimal_places] = \
                        decimal_info['decimal_places_count'].get(decimal_places, 0) + 1
                    decimal_info['max_decimal_places'] = max_decimal_places
                    
                except (ValueError, TypeError):
                    original_data.append(None)
                    blank_count += 1
        
        decimal_info['detected_decimal_places'] = max_decimal_places
        
        if blank_count > 0:
            st.warning(f"检测到 {blank_count} 个空白或无效数据，已自动过滤")
        
        return np.array(clean_data), original_data, blank_count, decimal_info

# =============================================
# 数据处理函数
# =============================================

def analyze_data():
    """分析手动输入的数据"""
    try:
        data_string = st.session_state.manual_data
        
        if not data_string or data_string.strip() == "":
            st.error("❌ 请输入数据")
            return
        
        calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
        
        is_valid, original_data, clean_data, blank_count, validation_report, decimal_info = \
            DataValidator.comprehensive_validation(data_string, calculation_scheme)
        
        if is_valid:
            st.session_state.processed_data = np.array(clean_data)
            st.session_state.original_data = original_data
            st.session_state.blank_count = blank_count
            st.session_state.validation_report = validation_report
            st.session_state.validation_passed = True
            st.session_state.decimal_info = decimal_info
            st.session_state.data_loaded = True
            
            if not st.session_state.data_history or st.session_state.data_history[-1] != data_string:
                st.session_state.data_history.append(data_string)
            
            st.success(f"✅ 数据验证通过！成功加载 {len(clean_data)} 个有效数据点")
            if blank_count > 0:
                st.warning(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
        else:
            st.error("❌ 数据验证失败，请检查输入格式")
            st.session_state.validation_passed = False
            
    except Exception as e:
        st.error(f"❌ 数据分析错误: {str(e)}")

def clear_data():
    """清除手动输入的数据"""
    st.session_state.manual_data = ""
    st.session_state.reset_counter += 1
    st.session_state.data_loaded = False
    st.rerun()

def undo_data():
    """撤销到上一次的数据状态"""
    if len(st.session_state.data_history) > 0:
        st.session_state.data_history.pop()
        if len(st.session_state.data_history) > 0:
            previous_data = st.session_state.data_history[-1]
            st.session_state.manual_data = previous_data
            try:
                analyze_data()
                st.success("✅ 已撤销到上一步")
            except:
                st.warning("⚠️ 撤销操作完成，但重新分析数据时出现问题")
        else:
            st.session_state.manual_data = ""
            st.session_state.data_loaded = False
            st.session_state.validation_passed = False
            st.success("✅ 已清除所有数据")
        
        st.rerun()
    else:
        st.warning("⚠️ 没有历史记录可撤销")

def analyze_two_column_data():
    """分析两列数据"""
    try:
        two_column_input = st.session_state.two_column_data
        calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
        
        label_data_pairs, valid_pairs, invalid_lines, decimal_info = \
            validate_two_column_data(two_column_input, calculation_scheme)
        
        if invalid_lines:
            st.error("❌ 数据格式错误:")
            for error in invalid_lines:
                st.error(f"  - {error}")
            return
            
        if not valid_pairs:
            st.error("❌ 未找到有效数据")
            return
        
        data = np.array([value for _, value in valid_pairs])
        labels = [label for label, _ in valid_pairs]
        
        st.session_state.label_data_pairs = label_data_pairs
        st.session_state.valid_pairs = valid_pairs
        st.session_state.processed_data = data
        st.session_state.original_labels = labels
        st.session_state.two_column_processed = True
        st.session_state.two_column_decimal_info = decimal_info
        
        st.success(f"✅ 成功加载 {len(data)} 个有效数据点")
        
    except Exception as e:
        st.error(f"❌ 两列数据分析错误: {str(e)}")

def clear_two_column_data():
    """清除两列数据"""
    st.session_state.two_column_data = ""
    st.session_state.two_column_processed = False
    st.session_state.label_data_pairs = []
    st.session_state.valid_pairs = []
    st.rerun()

# =============================================
# 辅助函数
# =============================================

def detect_decimal_places(data):
    """检测数据的小数位数"""
    if data is None or (hasattr(data, '__len__') and len(data) == 0):
        return 0
    
    max_decimal_places = 0
    
    try:
        data_array = np.asarray(data)
        for value in data_array:
            if isinstance(value, (int, float)) and not np.isnan(value):
                str_value = str(value)
                if 'e' in str_value.lower():
                    str_value = format(value, '.15f')
                if '.' in str_value:
                    decimal_part = str_value.split('.')[1].rstrip('0')
                    current_decimal_places = len(decimal_part)
                    max_decimal_places = max(max_decimal_places, current_decimal_places)
        return max_decimal_places
    except Exception:
        return 2

# =============================================
# 统计方法实现
# =============================================

def iterative_robust_algorithm(data, max_iterations=50, k=1.5, scheme="strict"):
    """迭代稳健统计法"""
    if data is None or len(data) == 0:
        return {
            'robust_mean': 0.0,
            'robust_std': 0.0,
            'clean_data': [],
            'outliers': [],
            'Z_scores_high_precision': [],
            'Z_scores_rounded': [],
            'formatted_Z_scores': [],
            'z_score_classifications': [],
            'iterations': 0,
            'converged': False,
            'lower_limit': 0.0,
            'upper_limit': 0.0,
            'history': [],
            'method_name': '迭代稳健统计法',
            'decimal_places': 0,
            'calculation_scheme': scheme,
            'formatting_note': "输入数据为空",
            'original_mean': 0.0,
            'original_std': 0.0
        }
    
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
    
    decimal_places = detect_decimal_places(data)
    
    if scheme == "presentation":
        formatted_X_star = round(X_star, decimal_places)
        formatted_S_star = round(S_star, 3)
        
        Z_scores_high_precision = (data - formatted_X_star) / formatted_S_star
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_X_star})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
        
        robust_mean = formatted_X_star
        robust_std = formatted_S_star
        
    else:
        formatted_X_star = X_star
        formatted_S_star = S_star
        
        Z_scores_high_precision = (data - X_star) / S_star
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
        
        robust_mean = X_star
        robust_std = S_star
    
    final_delta = k * robust_std
    lower_limit = robust_mean - final_delta
    upper_limit = robust_mean + final_delta
    
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    
    outliers_list = []
    clean_data_list = []
    
    for i, value in enumerate(data):
        if outliers_mask[i]:
            outliers_list.append(float(value))
        else:
            clean_data_list.append(float(value))
    
    safe_z_scores_high_precision = Z_scores_high_precision.tolist() if hasattr(Z_scores_high_precision, 'tolist') else [float(z) for z in Z_scores_high_precision]
    safe_z_scores_rounded = Z_scores_rounded.tolist() if hasattr(Z_scores_rounded, 'tolist') else [float(z) for z in Z_scores_rounded]
    
    formatted_Z_scores = format_z_scores(safe_z_scores_rounded)
    z_score_classifications = [classify_z_score(z) for z in safe_z_scores_rounded]
    
    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores_high_precision': safe_z_scores_high_precision,
        'Z_scores_rounded': safe_z_scores_rounded,
        'formatted_Z_scores': formatted_Z_scores,
        'z_score_classifications': z_score_classifications,
        'iterations': iteration,
        'converged': converged,
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'history': history,
        'method_name': '迭代稳健统计法',
        'decimal_places': decimal_places,
        'calculation_scheme': scheme,
        'formatting_note': formatting_note,
        'original_mean': float(X_star) if not np.isnan(X_star) else 0.0,
        'original_std': float(S_star) if not np.isnan(S_star) else 0.0
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
    
    decimal_places = detect_decimal_places(data)
    
    if scheme == "presentation":
        formatted_median = round(median, decimal_places)
        formatted_niqr = round(niqr, 3)
        
        Z_scores_high_precision = (data - formatted_median) / formatted_niqr
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_median})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
        
        robust_mean = formatted_median
        robust_std = formatted_niqr
        
    else:
        formatted_median = median
        formatted_niqr = niqr
        
        Z_scores_high_precision = (data - median) / niqr
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
        
        robust_mean = median
        robust_std = niqr
    
    safe_z_scores_high_precision = Z_scores_high_precision.tolist() if hasattr(Z_scores_high_precision, 'tolist') else [float(z) for z in Z_scores_high_precision]
    safe_z_scores_rounded = Z_scores_rounded.tolist() if hasattr(Z_scores_rounded, 'tolist') else [float(z) for z in Z_scores_rounded]

    formatted_Z_scores = format_z_scores(safe_z_scores_rounded)
    z_score_classifications = [classify_z_score(z) for z in safe_z_scores_rounded]

    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores_high_precision': safe_z_scores_high_precision,
        'Z_scores_rounded': safe_z_scores_rounded,
        'formatted_Z_scores': formatted_Z_scores,
        'z_score_classifications': z_score_classifications,
        'q1': float(q1) if not np.isnan(q1) else 0.0,
        'q3': float(q3) if not np.isnan(q3) else 0.0,
        'iqr': float(iqr) if not np.isnan(iqr) else 0.0,
        'niqr': float(niqr) if not np.isnan(niqr) else 0.0,
        'method_name': '四分位稳健统计法',
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'formatting_note': formatting_note,
        'calculation_scheme': scheme,
        'decimal_places': decimal_places,
        'original_mean': float(median) if not np.isnan(median) else 0.0,
        'original_std': float(niqr) if not np.isnan(niqr) else 0.0
    }

def z_score_calculation_algorithm(data, robust_mean, robust_std, scheme="strict"):
    """Z比分计算方法"""
    try:
        data_array = np.asarray(data, dtype=float)
        
        try:
            robust_mean_val = float(robust_mean)
        except ValueError:
            raise ValueError("稳健平均值格式错误，请输入有效的数字")
        
        try:
            robust_std_val = float(robust_std)
            if robust_std_val <= 0:
                raise ValueError("稳健标准差必须大于0")
        except ValueError:
            raise ValueError("稳健标准差格式错误，请输入有效的数字")
        
        decimal_places = detect_decimal_places(data_array)
        
        if scheme == "presentation":
            formatted_robust_mean = round(robust_mean_val, decimal_places)
            formatted_robust_std = round(robust_std_val, 3)
            
            Z_scores_high_precision = (data_array - formatted_robust_mean) / formatted_robust_std
            Z_scores_rounded = np.round(Z_scores_high_precision, 2)
            
            formatting_note = f"使用规范展示方案：稳健平均值({formatted_robust_mean})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
            
            robust_mean_display = formatted_robust_mean
            robust_std_display = formatted_robust_std
            
        else:
            formatted_robust_mean = robust_mean_val
            formatted_robust_std = robust_std_val
            
            Z_scores_high_precision = (data_array - robust_mean_val) / robust_std_val
            Z_scores_rounded = np.round(Z_scores_high_precision, 2)
            
            formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
            
            robust_mean_display = robust_mean_val
            robust_std_display = robust_std_val
        
        lower_limit = robust_mean_display - 3 * robust_std_display
        upper_limit = robust_mean_display + 3 * robust_std_display
        
        outliers_mask = (data_array < lower_limit) | (data_array > upper_limit)
        
        outliers_list = []
        clean_data_list = []
        
        for i, value in enumerate(data_array):
            if outliers_mask[i]:
                outliers_list.append(float(value))
            else:
                clean_data_list.append(float(value))
        
        safe_z_scores_high_precision = Z_scores_high_precision.tolist() if hasattr(Z_scores_high_precision, 'tolist') else [float(z) for z in Z_scores_high_precision]
        safe_z_scores_rounded = Z_scores_rounded.tolist() if hasattr(Z_scores_rounded, 'tolist') else [float(z) for z in Z_scores_rounded]
        
        formatted_Z_scores = format_z_scores(safe_z_scores_rounded)
        z_score_classifications = [classify_z_score(z) for z in safe_z_scores_rounded]
        
        return {
            'robust_mean': float(robust_mean_display),
            'robust_std': float(robust_std_display),
            'clean_data': clean_data_list,
            'outliers': outliers_list,
            'Z_scores_high_precision': safe_z_scores_high_precision,
            'Z_scores_rounded': safe_z_scores_rounded,
            'formatted_Z_scores': formatted_Z_scores,
            'z_score_classifications': z_score_classifications,
            'method_name': 'Z比分计算模块',
            'lower_limit': float(lower_limit),
            'upper_limit': float(upper_limit),
            'formatting_note': formatting_note,
            'calculation_scheme': scheme,
            'decimal_places': decimal_places,
            'original_mean': float(robust_mean_val),
            'original_std': float(robust_std_val)
        }
        
    except Exception as e:
        raise e

# =============================================
# Q/Hampel法实现
# =============================================

def perform_Q_estimate_corrected(x_data):
    """修正后的Q方法实现"""
    if isinstance(x_data, pd.Series):
        x_values = x_data.astype('float').values
    else:
        x_values = np.asarray(x_data, dtype=float)
    
    x_values = x_values[~np.isnan(x_values)]
    p = len(x_values)
    
    d_arr = []
    for i in range(p-1):
        for j in range(i+1, p):
            d = np.abs(x_values[i] - x_values[j])
            d_arr.append(d)
    
    d_arr = np.array(d_arr)
    H1_0 = np.mean(d_arr <= 0)
    discontinuity_points = np.unique(d_arr)
    H1_values = [np.mean(d_arr <= point) for point in discontinuity_points]
    
    G1_points = [0.0]
    G1_values = [0.0]
    
    for i, point in enumerate(discontinuity_points):
        if i == 0 and point > 0:
            G1_val = 0.5 * H1_values[i]
            G1_points.append(point)
            G1_values.append(G1_val)
        elif i >= 1:
            G1_val = 0.5 * (H1_values[i] + H1_values[i-1])
            G1_points.append(point)
            G1_values.append(G1_val)
    
    G1_points = np.array(G1_points)
    G1_values = np.array(G1_values)
    
    if len(G1_points) > 1:
        n_interp = 10000
        x_interp = np.linspace(G1_points[0], G1_points[-1], n_interp)
        G1_interp_func = interpolate.interp1d(
            G1_points, G1_values, 
            kind='linear', 
            bounds_error=False,
            fill_value=(G1_values[0], G1_values[-1]),
            assume_sorted=True
        )
        G1_interp = G1_interp_func(x_interp)
    else:
        x_interp = G1_points
        G1_interp = G1_values
    
    target_G1 = 0.25 + 0.75 * H1_0
    
    idx_above = np.where(G1_interp >= target_G1)[0]
    idx_below = np.where(G1_interp <= target_G1)[0]
    
    if len(idx_above) > 0 and len(idx_below) > 0:
        idx_high = idx_above[0]
        idx_low = idx_below[-1]
        
        if idx_high == idx_low:
            numerator = x_interp[idx_high]
        else:
            x1, x2 = x_interp[idx_low], x_interp[idx_high]
            y1, y2 = G1_interp[idx_low], G1_interp[idx_high]
            
            if abs(y2 - y1) > 1e-12:
                numerator = x1 + (x2 - x1) * (target_G1 - y1) / (y2 - y1)
            else:
                numerator = (x1 + x2) / 2
    else:
        if len(idx_above) == 0:
            numerator = x_interp[-1]
        else:
            numerator = x_interp[0]
    
    target_phi = 0.625 + 0.375 * H1_0
    target_phi = np.clip(target_phi, 0.001, 0.999)
    denominator = np.sqrt(2) * norm.ppf(target_phi)
    
    Q = numerator / denominator if denominator > 1e-12 else 0.0
    return Q

def hampel_robust_mean(data, max_iterations=50, tol=1e-8):
    """修正的Hampel稳健平均值计算方法"""
    if len(data) == 0:
        return 0.0
    
    x = np.median(data)
    
    for iteration in range(max_iterations):
        residuals = data - x
        mad = np.median(np.abs(residuals))
        
        if mad < 1e-12:
            return float(x)
        
        u = residuals / (1.4826 * mad)
        weights = np.ones_like(data)
        abs_u = np.abs(u)
        
        mask1 = (abs_u > 1.5) & (abs_u <= 3.0)
        weights[mask1] = 1.5 / abs_u[mask1]
        
        mask2 = (abs_u > 3.0) & (abs_u <= 4.5)
        weights[mask2] = (4.5 - abs_u[mask2]) / 1.5 * 0.5
        
        mask3 = abs_u > 4.5
        weights[mask3] = 0
        
        total_weight = np.sum(weights)
        if total_weight < 1e-12:
            return float(x)
        
        x_new = np.sum(weights * data) / total_weight
        
        if abs(x_new - x) < tol:
            return float(x_new)
        
        x = x_new
    
    return float(x)

def q_hampel_robust_algorithm(data, scheme="strict"):
    """基于Q方法和修正Hampel方法的稳健统计方法"""
    try:
        data_array = np.asarray(data, dtype=float)
        n = len(data_array)
        
        if n == 0:
            return _create_empty_result(scheme)
        elif n == 1:
            return _create_single_point_result(data_array[0], scheme)
        
        robust_std = perform_Q_estimate_corrected(data_array)
        robust_mean = hampel_robust_mean(data_array)
        
        lower_limit = robust_mean - 3 * robust_std
        upper_limit = robust_mean + 3 * robust_std
        
        outliers_mask = (data_array < lower_limit) | (data_array > upper_limit)
        clean_data = data_array[~outliers_mask].tolist()
        outliers = data_array[outliers_mask].tolist()
        
        if robust_std > 1e-12:
            Z_scores_high_precision = ((data_array - robust_mean) / robust_std).tolist()
            Z_scores_rounded = np.round(Z_scores_high_precision, 2).tolist()
        else:
            Z_scores_high_precision = [0.0] * n
            Z_scores_rounded = [0.0] * n
        
        return _format_q_hampel_results(
            data_array, robust_mean, robust_std, clean_data, outliers, 
            Z_scores_high_precision, Z_scores_rounded,
            lower_limit, upper_limit, scheme
        )
        
    except Exception as e:
        return _fallback_method(data, scheme, str(e))

def _format_q_hampel_results(data, robust_mean, robust_std, clean_data, outliers, 
                           Z_scores_high_precision, Z_scores_rounded,
                           lower_limit, upper_limit, scheme):
    """格式化Q/Hampel法结果"""
    decimal_places = detect_decimal_places(data)
    
    if scheme == "presentation":
        formatted_robust_mean = round(robust_mean, decimal_places)
        formatted_robust_std = round(robust_std, 3)
        
        if formatted_robust_std > 1e-12:
            Z_scores_high_precision = ((data - formatted_robust_mean) / formatted_robust_std).tolist()
            Z_scores_rounded = np.round(Z_scores_high_precision, 2).tolist()
        else:
            Z_scores_high_precision = [0.0] * len(data)
            Z_scores_rounded = [0.0] * len(data)
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_robust_mean})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。"
        
        display_mean = formatted_robust_mean
        display_std = formatted_robust_std
        
    else:
        formatting_note = "使用严格计算方案：保留完整计算精度。"
        display_mean = robust_mean
        display_std = robust_std
    
    formatted_Z_scores = format_z_scores(Z_scores_rounded)
    z_score_classifications = [classify_z_score(z) for z in Z_scores_rounded]
    
    return {
        'robust_mean': float(display_mean),
        'robust_std': float(display_std),
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores_high_precision': Z_scores_high_precision,
        'Z_scores_rounded': Z_scores_rounded,
        'formatted_Z_scores': formatted_Z_scores,
        'z_score_classifications': z_score_classifications,
        'method_name': 'Q/Hampel法',
        'lower_limit': float(lower_limit),
        'upper_limit': float(upper_limit),
        'weights': np.ones_like(data).tolist(),
        'iterations': 0,
        'formatting_note': formatting_note,
        'calculation_scheme': scheme,
        'decimal_places': decimal_places,
        'original_mean': float(robust_mean),
        'original_std': float(robust_std)
    }

def _create_empty_result(scheme):
    """创建空数据结果"""
    return {
        'robust_mean': 0.0,
        'robust_std': 0.0,
        'clean_data': [],
        'outliers': [],
        'Z_scores_high_precision': [],
        'Z_scores_rounded': [],
        'formatted_Z_scores': [],
        'z_score_classifications': [],
        'method_name': 'Q/Hampel法（空数据）',
        'lower_limit': 0.0,
        'upper_limit': 0.0,
        'weights': [],
        'iterations': 0,
        'formatting_note': "输入数据为空",
        'calculation_scheme': scheme,
        'decimal_places': 0,
        'original_mean': 0.0,
        'original_std': 0.0
    }

def _create_single_point_result(value, scheme):
    """创建单数据点结果"""
    return {
        'robust_mean': float(value),
        'robust_std': 0.0,
        'clean_data': [float(value)],
        'outliers': [],
        'Z_scores_high_precision': [0.0],
        'Z_scores_rounded': [0.0],
        'formatted_Z_scores': ["0.00"],
        'z_score_classifications': ["满意"],
        'method_name': 'Q/Hampel法（单数据点）',
        'lower_limit': float(value),
        'upper_limit': float(value),
        'weights': [1.0],
        'iterations': 0,
        'formatting_note': "只有一个数据点，无法计算标准差",
        'calculation_scheme': scheme,
        'decimal_places': detect_decimal_places([value]),
        'original_mean': float(value),
        'original_std': 0.0
    }

def _fallback_method(data, scheme, error_message=""):
    """回退方法 - 使用传统统计量"""
    data_array = np.asarray(data, dtype=float)
    mean_val = float(np.mean(data_array))
    std_val = float(np.std(data_array, ddof=1)) if len(data_array) > 1 else 0.0
    
    if std_val > 0:
        Z_scores_high_precision = ((data_array - mean_val) / std_val).tolist()
        Z_scores_rounded = np.round(Z_scores_high_precision, 2).tolist()
    else:
        Z_scores_high_precision = [0.0] * len(data_array)
        Z_scores_rounded = [0.0] * len(data_array)
    
    formatted_Z_scores = format_z_scores(Z_scores_rounded)
    z_score_classifications = [classify_z_score(z) for z in Z_scores_rounded]
    
    return {
        'robust_mean': mean_val,
        'robust_std': std_val,
        'clean_data': data_array.tolist(),
        'outliers': [],
        'Z_scores_high_precision': Z_scores_high_precision,
        'Z_scores_rounded': Z_scores_rounded,
        'formatted_Z_scores': formatted_Z_scores,
        'z_score_classifications': z_score_classifications,
        'method_name': '回退方法',
        'lower_limit': mean_val - 3 * std_val,
        'upper_limit': mean_val + 3 * std_val,
        'formatting_note': f"使用传统平均值和标准差（原方法失败：{error_message[:100]}）",
        'calculation_scheme': scheme,
        'decimal_places': detect_decimal_places(data_array),
        'original_mean': mean_val,
        'original_std': std_val
    }

# =============================================
# 结果显示组件
# =============================================

def display_method_specific_info(results, method):
    """显示各方法特有的统计信息"""
    if method == "四分位稳健统计法":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("下四分位数(Q1)", f"{results['q1']:.6f}")
        with col2:
            st.metric("上四分位数(Q3)", f"{results['q3']:.6f}")
        with col3:
            st.metric("四分位距(IQR)", f"{results['iqr']:.6f}")
        with col4:
            st.metric("标准化四分位距(NIQR)", f"{results['niqr']:.6f}")
    
    elif method == "Q/Hampel法":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("初始中位数", f"{results.get('initial_median', results['robust_mean']):.6f}")
        with col2:
            st.metric("MAD", f"{results.get('mad', 0):.6f}")

def display_core_results(results, method):
    """显示核心结果"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
    with col2:
        st.metric("稳健标准差", f"{results['robust_std']:.6f}")
    with col3:
        st.metric("离群值数量", len(results['outliers']))
    with col4:
        if 'iterations' in results:
            st.metric("迭代次数", results['iterations'])
        else:
            st.metric("计算方案", "规范展示" if results.get('calculation_scheme') == "presentation" else "严格计算")
    
    display_method_specific_info(results, method)

def display_z_score_analysis(results):
    """显示Z比分分析"""
    st.subheader("📊 Z比分分析")
    
    z_scores_data = results['Z_scores_rounded']
    if z_scores_data:
        try:
            z_scores = np.array(z_scores_data)
            z_scores_abs = np.abs(z_scores)
            satisfactory = np.sum(z_scores_abs <= 2)
            questionable = np.sum((z_scores_abs > 2) & (z_scores_abs < 3))
            unsatisfactory = np.sum(z_scores_abs >= 3)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sat_percent = satisfactory/len(z_scores)*100 if len(z_scores) > 0 else 0
                st.metric("满意 (|Z| ≤ 2)", f"{satisfactory} 个", f"{sat_percent:.1f}%")
            with col2:
                quest_percent = questionable/len(z_scores)*100 if len(z_scores) > 0 else 0
                st.metric("可疑 (2 < |Z| < 3)", f"{questionable} 个", f"{quest_percent:.1f}%")
            with col3:
                unsat_percent = unsatisfactory/len(z_scores)*100 if len(z_scores) > 0 else 0
                st.metric("不满意 (|Z| ≥ 3)", f"{unsatisfactory} 个", f"{unsat_percent:.1f}%")
                
        except Exception as e:
            st.warning(f"无法计算Z比分分类: {str(e)}")

def display_detailed_results(results, method, data):
    """显示详细结果"""
    with st.expander("📋 详细结果", expanded=False):
        st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
        
        if 'formatting_note' in results:
            st.info(f"💡 {results['formatting_note']}")
        
        if len(results['outliers']) > 0:
            outliers_list = results['outliers']
            if hasattr(outliers_list, '__iter__') and not isinstance(outliers_list, str):
                try:
                    outliers_list = [float(x) for x in outliers_list]
                    outliers_list = sorted(outliers_list)
                    
                    st.write(f"**离群值** ({len(outliers_list)}个):")
                    
                    robust_mean = results['robust_mean']
                    robust_std = results['robust_std']
                    
                    questionable_outliers = []
                    unsatisfactory_outliers = []
                    
                    for outlier in outliers_list:
                        z_score = abs((outlier - robust_mean) / robust_std) if robust_std > 0 else float('inf')
                        if 2 < z_score < 3:
                            questionable_outliers.append((outlier, z_score))
                        elif z_score >= 3:
                            unsatisfactory_outliers.append((outlier, z_score))
                    
                    if questionable_outliers:
                        st.write(f"  - 可疑离群值 (2<|Z|<3): {[f'{val[0]} (Z={val[1]:.2f})' for val in questionable_outliers]}")
                    
                    if unsatisfactory_outliers:
                        st.write(f"  - 不满意离群值 (|Z|≥3): {[f'{val[0]} (Z={val[1]:.2f})' for val in unsatisfactory_outliers]}")
                        
                except (ValueError, TypeError):
                    st.write("**离群值**: [无法显示]")
            else:
                st.write("**离群值**: 无")
        else:
            st.success("✅ **离群值**: 无检测到离群值")

def display_z_score_comparison_table(results, original_labels=None):
    """显示高精度和保留两位小数后Z比分数对比表格"""
    st.subheader("📊 Z比分数对比表格")
    
    n_points = len(results['Z_scores_high_precision'])
    
    if original_labels is None or len(original_labels) != n_points:
        labels = [f"数据点 {i+1:03d}" for i in range(n_points)]
    else:
        labels = original_labels
    
    comparison_data = []
    for i in range(n_points):
        high_precision_z = results['Z_scores_high_precision'][i]
        rounded_z = results['Z_scores_rounded'][i]
        classification = results['z_score_classifications'][i]
        
        comparison_data.append({
            '数据点': labels[i],
            '高精度Z比分数': f"{high_precision_z:.6f}",
            '保留两位小数Z比分数': f"{rounded_z:.2f}",
            '分类结果': classification
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.subheader("📈 Z比分数分类统计")
    classification_counts = Counter(results['z_score_classifications'])
    total_count = len(results['z_score_classifications'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        satisfactory_count = classification_counts.get('满意', 0)
        satisfactory_percent = (satisfactory_count / total_count * 100) if total_count > 0 else 0
        st.metric("满意 (|Z| ≤ 2)", f"{satisfactory_count} 个", f"{satisfactory_percent:.1f}%")
    
    with col2:
        questionable_count = classification_counts.get('可疑', 0)
        questionable_percent = (questionable_count / total_count * 100) if total_count > 0 else 0
        st.metric("可疑 (2 < |Z| < 3)", f"{questionable_count} 个", f"{questionable_percent:.1f}%")
    
    with col3:
        unsatisfactory_count = classification_counts.get('不满意', 0)
        unsatisfactory_percent = (unsatisfactory_count / total_count * 100) if total_count > 0 else 0
        st.metric("不满意 (|Z| ≥ 3)", f"{unsatisfactory_count} 个", f"{unsatisfactory_percent:.1f}%")
    
    return comparison_df

def create_z_score_chart(results, original_labels=None):
    """创建Z比分分布图表"""
    try:
        set_chinese_font()
        
        z_scores = results['Z_scores_rounded']
        classifications = results['z_score_classifications']
        if not z_scores or not classifications:
            return None
          
        n_points = len(z_scores)
        if original_labels is None or len(original_labels) != n_points:
            labels = [f"{i+1:03d}" for i in range(n_points)]
        else:
            labels = original_labels
        
        chart_data = pd.DataFrame({
            'Label': labels,
            'Z_Score': z_scores,
            'Classification': classifications
        })
        
        chart_data = chart_data.sort_values('Z_Score', ascending=False)
        chart_height = max(10, len(chart_data) * 0.4)
        fig, ax = plt.subplots(figsize=(14, chart_height))
        
        color_map = {
            '满意': '#00FF00',
            '可疑': '#FFA500',
            '不满意': '#FF0000',
            '未知': '#808080'
        }
        
        colors = [color_map.get(cat, '#808080') for cat in chart_data['Classification']]
        
        y_positions = range(len(chart_data))
        bars = ax.barh(
            y_positions, 
            chart_data['Z_Score'], 
            color=colors, 
            alpha=0.6,
            height=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        for i, (bar, z_value) in enumerate(zip(bars, chart_data['Z_Score'])):
            try:
                text_color = 'black'
                z_display = f"{z_value:.2f}"
                
                ax.text(
                    bar.get_width() + 0.05 * (1 if bar.get_width() >= 0 else -1), 
                    bar.get_y() + bar.get_height()/2, 
                    z_display, 
                    ha='left' if bar.get_width() >= 0 else 'right', 
                    va='center', 
                    fontsize=9, 
                    fontweight='bold',
                    color=text_color
                )
            except:
                continue
        
        ax.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Original Data ID', fontsize=14, fontweight='bold')
        ax.set_title('Z-Score Distribution (Sorted)', fontsize=18, fontweight='bold', pad=40)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['满意'], alpha=0.6, label='Satisfactory (|Z| ≤ 2)'),
            Patch(facecolor=color_map['可疑'], alpha=0.6, label='Questionable (2 < |Z| < 3)'),
            Patch(facecolor=color_map['不满意'], alpha=0.6, label='Unsatisfactory (|Z| ≥ 3)')
        ]
        
        ax.legend(
            handles=legend_elements, 
            title='Category', 
            title_fontsize=12, 
            fontsize=11, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.00), 
            ncol=3, 
            frameon=True
        )
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(chart_data['Label'])
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('white')
        ax.invert_yaxis()
        
        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"创建图表时发生错误: {str(e)}")
        return None

# =============================================
# 主UI界面
# =============================================

def main():
    """主函数"""
    st.set_page_config(
        page_title="统计宝 | 稳健统计分析工具 ",
        page_icon="📊",
        layout="wide"
    )
    
    initialize_session_state()
    
    # 添加CSS样式
    st.markdown("""
    <style>
    .main-header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .header-icon {
        margin-right: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 页面头部
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            icon = Image.open("stataid_cut edge.png")
            st.image(icon, width=200)
        except:
            st.image("📊", width=100)
    with col2:
        st.markdown("### **统计宝**")
        st.markdown("提供多种稳健统计分析方法，用于处理包含异常值的数据集。")
    
    # 侧边栏
    st.sidebar.header("⚙️ 分析设置")
    
    method = st.sidebar.selectbox(
        "选择统计方法:",
        ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法", "Z比分计算模块"],
        help="选择适合数据特征的稳健统计方法"
    )
    
    if method == "迭代稳健统计法":
        st.sidebar.subheader("迭代法参数")
        k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
        max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)
    
    st.sidebar.subheader("计算方案")
    calculation_scheme = st.sidebar.radio(
        "选择计算方案:",
        ["严格计算方案", "规范展示方案"],
        help="""
        严格计算方案：使用完整精度的计算结果，确保计算准确性
        规范展示方案：稳健平均值与原始数据小数位数一致，结果更规范但可能引入微小误差
        """,
        key="calculation_scheme_selector"
    )
    
    st.session_state.calculation_scheme = calculation_scheme
    
    st.sidebar.subheader("高级选项")
    show_scheme_comparison = st.sidebar.checkbox(
        "显示方案比较", 
        help="同时显示两种计算方案的结果对比",
        value=False
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📚 方法说明")
    
    method_descriptions = {
        "迭代稳健统计法": "通过迭代过程逐步修正异常值影响，收敛后得到稳健的统计估计。",
        "四分位稳健统计法": "以数据排序为基础，使用数据集中段50%的数据，崩溃点为25%，具有易于计算、操作简单的特点。",
        "Q/Hampel法": "结合Q方法计算的稳健标准差和Hampel方法计算的稳健平均值，具有较好的抗异常值干扰能力。",
        "Z比分计算模块": "使用用户提供的稳健统计量计算Z比分：Z比分 = (测试数据-稳健平均值)/稳健标准差"
    }
    
    st.sidebar.info(method_descriptions[method])
    
    if method == "Z比分计算模块":
        st.sidebar.info("💡 **注意**: Z比分计算参数请在主界面输入")
    
    # 数据输入方式
    st.markdown("### **数据输入方式:**")
    input_method = st.radio("", 
                           ["手动输入", "带编号数据输入", "文件上传", "示例数据"],
                           horizontal=True,
                           index=0,
                           label_visibility="collapsed")
    
    data = None
    
    # 手动输入
    if input_method == "手动输入":
        st.subheader("📝 手动输入数据")
        
        manual_input = st.text_area(
            "请输入数据（每行一个数值或用逗号分隔）:",
            value=st.session_state.manual_data,
            height=150,
            key=f"manual_input_{st.session_state.reset_counter}",
            help="支持空白数据自动忽略"
        )
        
        st.session_state.manual_data = manual_input
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("分析数据", use_container_width=True, type="primary"):
                analyze_data()
        
        with col2:
            if st.button("一键清除", use_container_width=True, type="secondary"):
                clear_data()
        
        with col3:
            undo_disabled = len(st.session_state.data_history) <= 1
            undo_label = f"↶ 撤销 ({len(st.session_state.data_history)-1}次可用)" if len(st.session_state.data_history) > 1 else "↶ 撤销 (无历史)"
            if st.button(undo_label, use_container_width=True, disabled=undo_disabled):
                undo_data()
        
        if st.session_state.validation_passed:
            with st.expander("📋 查看数据验证报告", expanded=True):
                for line in st.session_state.validation_report:
                    if line.startswith("❌"):
                        st.error(line)
                    elif line.startswith("⚠️"):
                        st.warning(line)
                    elif line.startswith("📊"):
                        st.write("**" + line + "**")
                    else:
                        st.write(line)
        
        if st.session_state.data_loaded and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            if st.session_state.blank_count > 0:
                st.warning(f"⚠️ 检测到 {st.session_state.blank_count} 个空白数据点，已忽略")
    
    # 带编号数据输入
    elif input_method == "带编号数据输入":
        st.subheader("📝 带编号数据输入")
        
        st.markdown("""
        **输入格式说明：**
        - 每行输入一个数据对，格式为：`标签, 数值`
        - 标签可以是任意字符串（如样本编号、名称等）
        - 数值必须是有效的数字
        - 示例：
            ```
            Sample_A, 54.4
            Sample_B, 54.6
            Control_1, 54.2
            ```
        """)
        
        two_column_input = st.text_area(
            "请输入标签和数值数据（每行一个数据对，用逗号分隔）:",
            value=st.session_state.two_column_data,
            height=200,
            key="two_column_input",
            help="格式：标签, 数值"
        )
        
        st.session_state.two_column_data = two_column_input
        
        if st.button("分析两列数据", type="primary", use_container_width=True):
            analyze_two_column_data()
        
        if st.button("清除两列数据", type="secondary", use_container_width=True):
            clear_two_column_data()
        
        if st.session_state.two_column_processed and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
    
    # 文件上传
    elif input_method == "文件上传":
        st.subheader("📁 上传数据文件")
        
        uploaded_file = st.file_uploader(
            "选择数据文件 (支持 CSV、TXT、Excel、JSON)", 
            type=['csv', 'txt', 'xlsx', 'xls', 'json'],
            help="支持多种文件格式：CSV、文本文件、Excel工作簿、JSON数据文件。空白数据会自动识别并忽略。"
        )
        
        if uploaded_file is not None:
            try:
                file_size = uploaded_file.size / 1024
                st.write(f"📄 **文件信息**: {uploaded_file.name} ({file_size:.1f} KB)")
                
                file_format = FileProcessor.detect_file_format(uploaded_file)
                st.write(f"🔍 **检测到的格式**: {file_format.upper()}")
                
                if file_format == 'excel':
                    df, sheet_name, all_sheets = FileProcessor.process_excel_file(uploaded_file)
                elif file_format == 'csv':
                    df, sheet_name, all_sheets = FileProcessor.process_csv_file(uploaded_file)
                elif file_format == 'json':
                    df, sheet_name, all_sheets = FileProcessor.process_json_file(uploaded_file)
                else:
                    df, sheet_name, all_sheets = FileProcessor.process_txt_file(uploaded_file)
                
                if df is not None:
                    clean_data, original_data, blank_count, decimal_info = FileProcessor.extract_data_from_dataframe(df, sheet_name)
                    
                    if clean_data is not None and len(clean_data) > 0:
                        validation_report = [
                            "✅ 文件格式验证通过",
                            f"✅ 成功从 '{sheet_name}' 提取数据",
                            f"📊 总数据点数: {len(original_data)}",
                            f"📈 有效数据数: {len(clean_data)}",
                            f"⚠️ 空白数据数: {blank_count}" if blank_count > 0 else "✅ 未发现空白数据",
                            f"📏 检测到的小数位数: {decimal_info['detected_decimal_places']}位"
                        ]
                        
                        calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
                        scheme_messages = DataValidator.validate_calculation_scheme_compatibility(
                            clean_data, calculation_scheme, decimal_info
                        )
                        validation_report.extend(scheme_messages)
                        
                        recommended_scheme, recommendation_reason = DataValidator.get_recommended_scheme(decimal_info)
                        validation_report.append(f"💡 推荐计算方案: {recommended_scheme} - {recommendation_reason}")
                        
                        st.session_state.file_processed_data = clean_data
                        st.session_state.file_original_data = original_data
                        st.session_state.file_blank_count = blank_count
                        st.session_state.file_validation_report = validation_report
                        st.session_state.file_validation_passed = True
                        st.session_state.file_decimal_info = decimal_info
                        
                        st.session_state.processed_data = clean_data
                        st.session_state.original_data = original_data
                        st.session_state.blank_count = blank_count
                        st.session_state.decimal_info = decimal_info
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ 文件验证通过！成功加载 {len(clean_data)} 个有效数据点")
                        if blank_count > 0:
                            st.warning(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
                        
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
                        
                        data = clean_data
                    
                    else:
                        st.error("❌ 无法从文件中提取有效数据")
                
                else:
                    st.error("❌ 文件数据验证失败或没有有效数据")
            
            except Exception as e:
                st.error(f"❌ 文件处理错误: {str(e)}")
                st.info("💡 请确保文件格式正确且包含有效的数值数据")
        
        if st.session_state.file_validation_passed and st.session_state.file_processed_data is not None:
            data = st.session_state.file_processed_data
    
    # 示例数据
    else:
        st.subheader("🎯 示例数据分析")
        example_data = np.array([
            54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
            54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
            54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
            54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
            54.5, 55.9, 53.2, 54.6
        ])
        
        if st.button("使用示例数据进行分析", type="primary"):
            data = example_data
            st.session_state.processed_data = example_data
            st.session_state.original_data = example_data.tolist()
            st.session_state.data_loaded = True
            st.session_state.blank_count = 0
            
            st.success(f"✅ 示例数据已加载，包含 {len(example_data)} 个测量值")
            st.rerun()
        
        data = example_data
        st.session_state.original_data = example_data.tolist()
        st.session_state.blank_count = 0
    
    # Z比分计算模块的参数输入
    if method == "Z比分计算模块":
        st.markdown("---")
        st.subheader("🔢 Z比分计算参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            robust_mean_input = st.text_input(
                "稳健平均值:",
                value=st.session_state.robust_mean_input,
                help="请输入稳健平均值",
                key="robust_mean_input"
            )
            st.session_state.robust_mean_input = robust_mean_input
        
        with col2:
            robust_std_input = st.text_input(
                "稳健标准差:",
                value=st.session_state.robust_std_input,
                help="请输入稳健标准差",
                key="robust_std_input"
            )
            st.session_state.robust_std_input = robust_std_input
        
        st.info("""
        **Z比分计算公式：**
        ```
        Z比分 = (测试数据 - 稳健平均值) / 稳健标准差
        ```
        请确保输入的稳健统计量准确无误。
        """)
    
    # 根据输入方式设置data变量
    if input_method == "手动输入":
        if st.session_state.data_loaded and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
    elif input_method == "带编号数据输入":
        if st.session_state.two_column_processed and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
    elif input_method == "文件上传":
        if st.session_state.file_validation_passed and st.session_state.file_processed_data is not None:
            data = st.session_state.file_processed_data
    
    # 执行分析
    if data is not None and len(data) > 0:
        try:
            if isinstance(data, list):
                data = np.array(data)
            elif not isinstance(data, np.ndarray):
                st.error("❌ 数据格式无效")
                st.stop()
            
            if len(data) == 0:
                st.error("❌ 没有有效数据可供分析")
                st.stop()
            
            st.markdown("---")
            st.subheader(f"📈 {method}分析结果")
            
            calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
            scheme_display = "规范展示方案" if calculation_scheme == "规范展示方案" else "严格计算方案"
            st.info(f"当前使用: **{scheme_display}**")
            
            if method == "Z比分计算模块":
                try:
                    robust_mean_val = float(st.session_state.robust_mean_input)
                    robust_std_val = float(st.session_state.robust_std_input)
                    
                    if robust_std_val <= 0:
                        st.error("❌ 稳健标准差必须大于0")
                        st.stop()
                    
                    st.success(f"✅ 使用参数: 稳健平均值 = {robust_mean_val}, 稳健标准差 = {robust_std_val}")
                    
                except ValueError:
                    st.error("❌ 稳健统计量格式错误，请输入有效的数字")
                    st.stop()
                except KeyError:
                    st.error("❌ 请先输入稳健统计量")
                    st.stop()
            
            with st.spinner(f"正在执行{method}分析..."):
                scheme_param = "presentation" if calculation_scheme == "规范展示方案" else "strict"
                
                if method == "迭代稳健统计法":
                    results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme=scheme_param)
                elif method == "四分位稳健统计法":
                    results = quartile_robust_algorithm(data, scheme=scheme_param)
                elif method == "Q/Hampel法":
                    results = q_hampel_robust_algorithm(data, scheme=scheme_param)
                elif method == "Z比分计算模块":
                    results = z_score_calculation_algorithm(data, robust_mean_val, robust_std_val, scheme=scheme_param)
            
            # 显示计算方案说明
            with st.expander("ℹ️ 计算方案说明", expanded=True):
                st.info(results['formatting_note'])
                st.success("💡 **Z比分处理说明**: Z比分在计算过程中保持完整精度，展示和导出时统一格式化为两位小数。分类基于保留两位小数后的Z比分数进行计算。")
            
            # 显示核心结果
            display_core_results(results, method)
            
            # 显示Z比分分析
            display_z_score_analysis(results)
            
            # 显示详细结果
            display_detailed_results(results, method, data)
            
            # 数据可视化
            st.subheader("📊 数据可视化")
            
            original_labels = None
            if input_method == "带编号数据输入" and st.session_state.label_data_pairs:
                original_labels = [pair[0] for pair in st.session_state.valid_pairs]
            else:
                n_points = len(data)
                original_labels = [f"{i+1:03d}" for i in range(n_points)]
            
            # 显示Z比分对比表格
            comparison_df = display_z_score_comparison_table(results, original_labels)
            
            # 显示Z比分图表
            fig = create_z_score_chart(results, original_labels)
            if fig is not None:
                st.pyplot(fig)
            
            # 方案比较功能
            if show_scheme_comparison and method != "Z比分计算模块":
                st.markdown("---")
                st.subheader("🔍 计算方案对比")
                
                with st.spinner("正在计算方案对比..."):
                    if method == "迭代稳健统计法":
                        strict_results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme="strict")
                        presentation_results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme="presentation")
                    elif method == "四分位稳健统计法":
                        strict_results = quartile_robust_algorithm(data, scheme="strict")
                        presentation_results = quartile_robust_algorithm(data, scheme="presentation")
                    else:
                        strict_results = q_hampel_robust_algorithm(data, scheme="strict")
                        presentation_results = q_hampel_robust_algorithm(data, scheme="presentation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**严格计算方案**")
                    st.write(f"稳健平均值: {strict_results['robust_mean']:.6f}")
                    st.write(f"稳健标准差: {strict_results['robust_std']:.6f}")
                    st.write(f"离群值数量: {len(strict_results['outliers'])}")
                
                with col2:
                    st.write("**规范展示方案**")
                    st.write(f"稳健平均值: {presentation_results['robust_mean']}")
                    st.write(f"稳健标准差: {presentation_results['robust_std']:.3f}")
                    st.write(f"离群值数量: {len(presentation_results['outliers'])}")
                
                st.info("""
                **方案差异说明:**
                - **严格计算方案**: 使用完整计算精度，确保计算准确性
                - **规范展示方案**: 稳健平均值与原始数据小数位数一致，结果更规范但可能引入微小误差
                - **Z比分处理**: Z比分的计算数据精度不同，计算方法相同，展示时统一格式化为两位小数
                """)
            
            # 导出结果模块
            st.subheader("💾 导出结果")
            
            def format_number(value, decimal_places):
                """根据小数位数格式化数字"""
                if value is None or pd.isna(value):
                    return None
                if decimal_places == 0:
                    return int(value)
                return round(value, decimal_places)
            
            # 根据输入方式选择正确的数据源
            if input_method == "文件上传":
                current_original_data = st.session_state.file_original_data
                current_decimal_info = st.session_state.file_decimal_info
                current_blank_count = st.session_state.file_blank_count
            elif input_method == "带编号数据输入":
                current_original_data = [value for _, value in st.session_state.valid_pairs]
                current_decimal_info = st.session_state.two_column_decimal_info
                current_blank_count = 0
            else:
                current_original_data = st.session_state.original_data
                current_decimal_info = st.session_state.decimal_info
                current_blank_count = st.session_state.blank_count
            
            detected_decimal_places = results.get('decimal_places', 2)
            if current_decimal_info and 'detected_decimal_places' in current_decimal_info:
                detected_decimal_places = current_decimal_info['detected_decimal_places']
            
            if detected_decimal_places is None:
                detected_decimal_places = 2
            
            # 创建结果DataFrame
            result_data = []
            
            if input_method == "带编号数据输入" and st.session_state.label_data_pairs:
                valid_data_count = 0
                for label, value in st.session_state.label_data_pairs:
                    z_score = None
                    if value is not None and valid_data_count < len(results['formatted_Z_scores']):
                        z_score = results['formatted_Z_scores'][valid_data_count]
                        classification = results['z_score_classifications'][valid_data_count]
                        valid_data_count += 1
                    
                    formatted_value = format_number(value, detected_decimal_places)
                    formatted_z_score = format_z_score_display(z_score)
                    
                    result_data.append({
                        '标签原始标号': label,
                        '输入数据': formatted_value,
                        'Z比分数': formatted_z_score,
                        '分类结果': classification
                    })
                
                total_data_count = len(st.session_state.label_data_pairs)
                blank_data_count = sum(1 for _, value in st.session_state.label_data_pairs if value is None)
                actual_analyzable_count = total_data_count - blank_data_count
                
            else:
                valid_data_count = 0
                
                if current_original_data:
                    for i, value in enumerate(current_original_data):
                        original_label = f"{str(i+1).zfill(3)}"
                        
                        if value is not None:
                            z_score = results['formatted_Z_scores'][valid_data_count] if valid_data_count < len(results['formatted_Z_scores']) else None
                            classification = results['z_score_classifications'][valid_data_count] if valid_data_count < len(results['z_score_classifications']) else "未知"
                            formatted_value = format_number(value, detected_decimal_places)
                            formatted_z_score = format_z_score_display(z_score)
                            
                            result_data.append({
                                '标签原始标号': original_label,
                                '输入数据': formatted_value,
                                'Z比分数': formatted_z_score,
                                '分类结果': classification
                            })
                            valid_data_count += 1
                        else:
                            result_data.append({
                                '标签原始标号': original_label,
                                '输入数据': None,
                                'Z比分数': "",
                                '分类结果': ""
                            })
                    
                    total_data_count = len(current_original_data)
                    blank_data_count = current_blank_count
                    actual_analyzable_count = len(data)
                else:
                    for i, value in enumerate(data):
                        original_label = f"{str(i+1).zfill(3)}"
                        z_score = results['formatted_Z_scores'][i] if i < len(results['formatted_Z_scores']) else None
                        classification = results['z_score_classifications'][i] if i < len(results['z_score_classifications']) else "未知"
                        
                        formatted_value = format_number(value, detected_decimal_places)
                        formatted_z_score = format_z_score_display(z_score)
                        
                        result_data.append({
                            '标签原始标号': original_label,
                            '输入数据': formatted_value,
                            'Z比分数': formatted_z_score,
                            '分类结果': classification
                        })
                    
                    total_data_count = len(data)
                    blank_data_count = 0
                    actual_analyzable_count = len(data)
            
            result_df = pd.DataFrame(result_data)
            
            if len(result_data) != len(results['formatted_Z_scores']):
                st.warning(f"⚠️ 数据数量不匹配: 结果数据({len(result_data)}) vs Z分数({len(results['formatted_Z_scores'])})")
                if len(result_data) > len(results['formatted_Z_scores']):
                    result_data = result_data[:len(results['formatted_Z_scores'])]
                else:
                    while len(result_data) < len(results['formatted_Z_scores']):
                        result_data.append({
                            '标签原始标号': f"{str(len(result_data)+1).zfill(3)}",
                            '输入数据': None,
                            'Z比分数': "",
                            '分类结果': ""
                        })
            
            result_df = pd.DataFrame(result_data)
            
            # 计算统计量
            stats_data = {
                '统计量名称': ['总数据数', '实际可分析数据数', '空白数据数', '指定值', '能力评定标准差', '最小值', '最大值', '极差'],
                '数值': [
                    total_data_count,
                    actual_analyzable_count,
                    blank_data_count,
                    format_number(results['robust_mean'], detected_decimal_places),
                    format_number(results['robust_std'], 3),
                    format_number(np.min(data), detected_decimal_places) if len(data) > 0 else 0,
                    format_number(np.max(data), detected_decimal_places) if len(data) > 0 else 0,
                    format_number(np.max(data) - np.min(data), detected_decimal_places) if len(data) > 0 else 0
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            
            # 显示预览
            st.write("**导出数据预览:**")
            st.dataframe(result_df, use_container_width=True)
            
            # 验证数据一致性
            st.write(f"**数据一致性验证:**")
            st.write(f"- 原始数据点数: {total_data_count}")
            st.write(f"- 有效分析数据: {actual_analyzable_count}")
            st.write(f"- 空白数据数: {blank_data_count}")
            st.write(f"- Z比分数量: {len(results['formatted_Z_scores'])}")
            st.write(f"- 导出数据行数: {len(result_df)}")
            
            if total_data_count != len(results['formatted_Z_scores']):
                st.error("❌ 数据数量不匹配！请检查数据处理逻辑。")
            else:
                st.success("✅ 数据一致性验证通过")
            
            st.write("**统计量摘要:**")
            st.dataframe(stats_df, use_container_width=True)
            
            # 创建文本报告
            scheme_text = "严格计算方案" if calculation_scheme == "严格计算方案" else "规范展示方案"
            report = f"""                
{method}分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
工具版本: 稳健统计分析工具 (Robust Statistical Analysis Tool)
计算方案: {scheme_text}
数据小数位数: {detected_decimal_places}位（基于输入数据的最大小数位数）

计算方案说明:
--------
为提升展示结果规范性，使用了四舍五入的稳健平均值和标准差来计算Z比分。
稳健平均值与原始数据保持相同的小数位数({detected_decimal_places}位)，
稳健标准差保留3位小数，Z比分在计算过程中保持完整精度，仅在展示和导出时统一格式化为两位小数。

数据概览:
--------
总数据点数: {total_data_count}
实际可分析数据数: {actual_analyzable_count}
空白数据数: {blank_data_count}

数据表格:
--------
标签原始标号\t输入数据\tZ比分数\t分类结果
"""
            
            for i in range(len(result_df)):
                row = result_df.iloc[i]
                if pd.isna(row['输入数据']):
                    input_data = ""
                else:
                    if detected_decimal_places == 0:
                        input_data = f"{int(row['输入数据'])}"
                    else:
                        input_data = f"{row['输入数据']:.{detected_decimal_places}f}"
                
                z_score = row['Z比分数']
                classification = row['分类结果']
                report += f"{row['标签原始标号']}\t{input_data}\t{z_score}\t{classification}\n"
            
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
            elif method == "Q/Hampel法":
                report += f"""
Q/Hampel统计量:
--------------
初始中位数: {results.get('initial_median', results['robust_mean']):.6f}
MAD值: {results.get('mad', 0):.6f}
迭代次数: {results.get('iterations', 0)}
收敛状态: {'是' if results.get('converged', True) else '否'}
"""
            
            if 'iterations' in results:
                report += f"迭代次数: {results['iterations']}\n"
            
            z_scores_abs = np.abs(results['Z_scores_rounded'])
            satisfactory = np.sum(z_scores_abs <= 2)
            questionable = np.sum((z_scores_abs > 2) & (z_scores_abs < 3))
            unsatisfactory = np.sum(z_scores_abs >= 3)
            
            report += f"""
Z比分数分类（仅有效数据）:
-------------------------
满意 (|Z| ≤ 2): {satisfactory} 个数据点
可疑 (2 < |Z| < 3): {questionable} 个数据点  
不满意 (|Z| ≥ 3): {unsatisfactory} 个数据点

离群值列表:
----------
"""
            
            if len(results['outliers']) > 0:
                outliers_list = [f"{float(x):.{detected_decimal_places}f}" for x in sorted(results['outliers'])]
                report += f"{', '.join(outliers_list)}"
            else:
                report += "无"
            
            # 创建多格式导出选项
            export_col1, export_col2, export_col3, export_col4 = st.columns(4)
            
            with export_col1:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    result_df.to_excel(writer, sheet_name='分析数据', index=False)
                    stats_df.to_excel(writer, sheet_name='统计摘要', index=False)
                    
                    detail_data = {
                        '项目': ['分析方法', '总数据点数', '实际可分析数据数', '空白数据数', 
                               '稳健平均值', '稳健标准差', '离群值数量', '正常值下限', '正常值上限',
                               '数据小数位数'],
                        '数值': [method, total_data_count, actual_analyzable_count, blank_data_count,
                               format_number(results['robust_mean'], detected_decimal_places),
                               format_number(results['robust_std'], 3),
                               len(results['outliers']), 
                               format_number(results['lower_limit'], detected_decimal_places),
                               format_number(results['upper_limit'], detected_decimal_places),
                               detected_decimal_places]
                    }
                    pd.DataFrame(detail_data).to_excel(writer, sheet_name='详细信息', index=False)
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 下载Excel",
                    data=excel_buffer,
                    file_name=f"{method}_分析结果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="下载Excel工作簿，包含分析数据和统计摘要"
                )
            
            with export_col2:
                export_data = {
                    "metadata": {
                        "export_time": pd.Timestamp.now().isoformat(),
                        "analysis_method": method,
                        "software": "稳健统计分析系统",
                        "data_summary": {
                            "total_data_points": total_data_count,
                            "actual_analyzable_data": actual_analyzable_count,
                            "blank_data_points": blank_data_count,
                            "decimal_places": detected_decimal_places
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
                    help="下载JSON格式的分析结果和数据"
                )
            
            with export_col3:
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载CSV",
                    data=csv_data,
                    file_name=f"{method}_分析结果.csv",
                    mime="text/csv",
                    help="下载CSV格式的分析结果表格"
                )
            
            with export_col4:
                st.download_button(
                    label="📥 下载报告",
                    data=report,
                    file_name=f"{method}_分析报告.txt",
                    mime="text/plain",
                    help="下载文本格式的详细分析报告"
                )
            
            # 图表下载功能
            st.subheader("📥 下载图表")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
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
            
            with chart_col2:
                if 'fig' in locals():
                    buffer_pdf = io.BytesIO()
                    fig.savefig(buffer_pdf, format="pdf", bbox_inches="tight")
                    buffer_pdf.seek(0)
                    st.download_button(
                        label="📥 下载PDF图表",
                        data=buffer_pdf,
                        file_name=f"z_score_chart_{method}.pdf",
                        mime="application/pdf"
                    )
            
            # 添加说明
            st.info(f"💡 **小数位数说明**: 导出的数据使用 {detected_decimal_places} 位小数（基于输入数据的最大小数位数）。Z比分统一格式化为两位小数，空白数据会保留标签但数据为空。")
            
            st.info("""
            **注意**: 分类基于保留两位小数后的Z比分数进行计算。
            """)
            
        except Exception as e:
            st.error(f"❌ 统计分析过程中发生错误: {str(e)}")
            st.info("💡 这可能是因为数据特征不适合所选的分析方法，请尝试其他统计方法或检查数据质量")
    
    else:
        st.info("👆 请先输入或上传数据以开始分析")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    **Z比分分类标准:**
    - **满意**: |Z| ≤ 2
    - **可疑**: 2 < |Z| < 3  
    - **不满意**: |Z| ≥ 3
    """)

# =============================================
# 运行应用
# =============================================

if __name__ == "__main__":
    main()

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