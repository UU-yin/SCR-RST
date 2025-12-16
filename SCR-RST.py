import streamlit as st
import numpy as np
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import json
from scipy import stats
from scipy.stats import norm
from scipy import interpolate
import matplotlib as mpl
import matplotlib.font_manager as fm
import base64
from io import BytesIO

# 设置中文字体
def set_chinese_font():
    """设置中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

# =============================================
# Z比分分类和格式化函数
# =============================================

def classify_z_score(z_score):
    """
    根据Z比分进行分类
    满意 (|Z| ≤ 2), 可疑(2 < |Z| < 3), 不满意 (|Z| ≥ 3)
    """
    try:
        z_abs = abs(float(z_score))
        if z_abs <= 2:
            return "满意"
        elif 2 < z_abs < 3:
            return "可疑"
        else:  # z_abs >= 3
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
# 数据验证和错误处理模块
# =============================================

class DataValidator:
    """数据验证器类 - 扩展支持计算方案验证"""
    
    @staticmethod
    def validate_numeric_string_with_blanks(data_string):
        """
        验证数值字符串格式，支持空白数据
        返回: (is_valid, original_data, clean_data, blank_count, error_message, decimal_info)
        """
        
        # 添加输入验证
        if data_string is None:
            return False, [], [], 0, "输入数据为None", {}

        if not isinstance(data_string, str):
            return False, [], [], 0, f"输入数据类型错误: {type(data_string)}，应为字符串", {}
        
        if not data_string or data_string.strip() == "":
            return False, [], [], 0, "输入数据不能为空", {}
        
        # 清理和分割数据
        lines = data_string.strip().split('\n')
        original_data = []  # 包含空白值的原始数据
        clean_data = []     # 清理后的有效数据
        blank_positions = [] # 空白数据的位置
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': 0
        }
        
        line_num = 0
        max_decimal_places = 0  # 改为记录最大小数位数
        
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
                        
                        # 分析小数位数
                        str_value = str(value)
                        if '.' in str_value:
                            decimal_part = str_value.split('.')[1]
                            # 去除末尾的零
                            decimal_part = decimal_part.rstrip('0')
                            decimal_places = len(decimal_part)
                        else:
                            decimal_places = 0
                        
                        # 更新最大小数位数
                        max_decimal_places = max(max_decimal_places, decimal_places)
                        
                        # 统计小数位数分布（保留用于其他分析）
                        decimal_info['decimal_places_count'][decimal_places] = \
                            decimal_info['decimal_places_count'].get(decimal_places, 0) + 1
                        decimal_info['max_decimal_places'] = max_decimal_places
                        
                        # 检查小数位数一致性
                        if decimal_info.get('previous_decimal_places') is not None and decimal_info['previous_decimal_places'] != decimal_places:
                            decimal_info['consistent_decimals'] = False
                        decimal_info['previous_decimal_places'] = decimal_places
                        
                    except ValueError:
                        return False, [], [], 0, f"数据格式错误: 第{line_num}行 '{item}' 不是有效的数字", {}
                else:
                    # 空白数据，记录位置并添加None作为占位符
                    original_data.append(None)
                    blank_positions.append((line_num, col_num))
        
        # 确定检测到的小数位数 - 使用最大小数位数
        decimal_info['detected_decimal_places'] = max_decimal_places
        
        blank_count = len(blank_positions)
        return True, original_data, clean_data, blank_count, "数据格式验证通过", decimal_info
                           
    @staticmethod
    def validate_calculation_scheme_compatibility(data_array, calculation_scheme, decimal_info):
        """
        验证计算方案与数据的兼容性
        """
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
        """
        根据数据特征推荐计算方案
        """
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
        blank_count = 0
        
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
        
        # 7. 小数位数统计（修改部分）
        if decimal_info['decimal_places_count']:
            decimal_stats = ", ".join([f"{places}位({count}个)" for places, count in decimal_info['decimal_places_count'].items()])
            validation_report.append(f"📊 数据小数位数分布: {decimal_stats}")
            validation_report.append(f"📏 使用的小数位数: {decimal_info['detected_decimal_places']}位（基于最大小数位数）")
            validation_report.append(f"🔍 小数位数一致性: {'是' if decimal_info['consistent_decimals'] else '否'}")
            
            # 添加关于小数位数选择的说明
            if not decimal_info['consistent_decimals']:
                validation_report.append("⚠️  检测到数据中小数位数不一致，将使用最大小数位数作为输出格式标准")
        else:
            validation_report.append("📊 数据小数位数: 均为整数")
            validation_report.append("📏 使用的小数位数: 0位（整数格式）")
        
        # 8. 异常值检测
        try:
            outliers, outliers_info = DataValidator.detect_potential_outliers(clean_data)
            if outliers_info:
                validation_report.append(f"⚠️ {outliers_info[0]}")
                if len(outliers) > 0:
                    # 确保 outliers 是列表且包含数值
                    if hasattr(outliers, '__iter__') and not isinstance(outliers, str):
                        # 安全地格式化异常值
                        try:
                            formatted_outliers = [f'{float(x):.4f}' for x in outliers]
                            validation_report.append(f"   异常值: {', '.join(formatted_outliers)}")
                        except (ValueError, TypeError):
                            validation_report.append("   异常值: [格式错误]")
                    else:
                        validation_report.append("   异常值: [无法显示]")
            else:
                validation_report.append("✅ 未发现明显异常值")
        except Exception as e:
            validation_report.append(f"⚠️ 异常值检测失败: {str(e)}")
        
        # 9. 数据统计信息（包含空白数据信息）
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
        """检测潜在异常值 - 完全重写版本"""
        try:
            if len(data_array) < 3:
                return [], ["数据点不足，无法进行异常值检测"]
            
            # 确保数据是numpy数组并处理可能的类型问题
            if not isinstance(data_array, np.ndarray):
                data_array = np.array(data_array)
            
            # 确保数据类型是数值型
            if not np.issubdtype(data_array.dtype, np.number):
                return [], ["数据包含非数值类型，无法进行异常值检测"]
            
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 使用更安全的方法获取异常值
            outliers_list = []
            for value in data_array:
                if value < lower_bound or value > upper_bound:
                    outliers_list.append(float(value))  # 明确转换为Python浮点数
            
            if len(outliers_list) > 0:
                return outliers_list, [f"检测到 {len(outliers_list)} 个潜在异常值（基于IQR方法）"]
            else:
                return [], ["未发现明显异常值"]
                
        except Exception as e:
            # 如果出现任何错误，返回空列表和错误信息
            return [], [f"异常值检测过程中发生错误: {str(e)}"]

# =============================================
# 两列数据输入验证函数
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
        max_decimal_places = 0  # 记录最大小数位数
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
                        
                        # 分析小数位数
                        str_value = str(value)
                        if '.' in str_value:
                            decimal_part = str_value.split('.')[1].rstrip('0')
                            decimal_places = len(decimal_part)
                        else:
                            decimal_places = 0
                        
                        # 更新最大小数位数
                        max_decimal_places = max(max_decimal_places, decimal_places)
                        
                        # 统计小数位数
                        decimal_info['decimal_places_count'][decimal_places] = \
                            decimal_info['decimal_places_count'].get(decimal_places, 0) + 1
                        decimal_info['max_decimal_places'] = max_decimal_places
                        
                        # 检查小数位数一致性
                        if previous_decimal_places is not None and previous_decimal_places != decimal_places:
                            decimal_info['consistent_decimals'] = False
                        previous_decimal_places = decimal_places
                        
                    except ValueError:
                        invalid_lines.append(f"第{i+1}行: '{value_str}' 不是有效的数字")
                else:
                    invalid_lines.append(f"第{i+1}行: 格式错误，应为'标签 数值'或'标签,数值'，当前内容: '{line}'")
        
        # 确定检测到的小数位数 - 使用最大小数位数
        decimal_info['detected_decimal_places'] = max_decimal_places
        
        return label_data_pairs, valid_pairs, invalid_lines, decimal_info
        
    except Exception as e:
        return [], [], [f"数据处理错误: {str(e)}"], {}

# =============================================
# 文件格式处理模块
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
                df = pd.DataFrame(data)
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
                
                if selected_key:
                    if isinstance(data[selected_key], list):
                        df = pd.DataFrame(data[selected_key])
                        return df, f"JSON字段: {selected_key}", available_keys
                    elif isinstance(data[selected_key], dict):
                        # 如果是嵌套对象，进一步选择
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
        """处理文本文件 - 支持多列数据"""
        try:
            content = uploaded_file.read().decode('utf-8')
            
            # 尝试多种分隔符解析文本文件
            separators = [',', '\t', ';', '|', ' ']
            df = None
            
            for sep in separators:
                try:
                    # 尝试使用当前分隔符解析
                    df = pd.read_csv(io.StringIO(content), sep=sep, na_filter=True, engine='python')
                    # 如果成功解析且有多个列，使用这个分隔符
                    if len(df.columns) > 1:
                        st.info(f"检测到文本文件使用分隔符: '{sep}'")
                        break
                except:
                    continue
            
            # 如果没有成功解析，使用默认的逗号分隔符
            if df is None:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=',', na_filter=True)
                except:
                    # 如果还是失败，将整个内容作为一列处理
                    lines = content.strip().split('\n')
                    data = []
                    for line in lines:
                        # 尝试提取数字
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
        """
        从DataFrame中提取数值数据 - 统一支持空白数据处理和多列选择
        返回: (clean_data, original_data, blank_count, decimal_info)
        """
        st.info(f"正在从 '{sheet_name}' 中提取数据")
        
        # 显示数据预览
        st.write("**数据预览:**")
        st.dataframe(df.head(), use_container_width=True)
        
        original_data = []  # 包含空白值的原始数据
        clean_data = []     # 清理后的有效数据
        blank_count = 0     # 空白数据计数
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': 0
        }
        max_decimal_places = 0
        
        # 数据提取逻辑 - 支持多列选择
        if len(df.columns) == 1:
            # 如果只有一列，直接使用
            data_column = df.iloc[:, 0]
            st.write(f"使用唯一列: {df.columns[0]}")
            
        else:
            # 多列时让用户选择
            st.write("检测到多列数据，请选择包含数值数据的列:")
            
            # 显示各列的数据类型和前几个值
            col_info = []
            for col in df.columns:
                sample_values = df[col].dropna().head(3).tolist()
                dtype = df[col].dtype
                col_info.append(f"{col} (类型: {dtype}, 样例: {sample_values})")
            
            selected_column = st.selectbox(
                "选择数据列:", 
                df.columns.tolist(),
                format_func=lambda x: f"{x} (类型: {df[x].dtype}, 样例: {df[x].dropna().head(3).tolist()})",
                key=f"column_selector_{hash(str(df.columns))}"  # 使用唯一的key
            )
            
            if selected_column:
                data_column = df[selected_column]
                st.success(f"已选择列: {selected_column}")
            else:
                st.error("请选择数据列")
                return None, [], 0, decimal_info
        
        # 在处理每个数值时，添加小数位数检测
        for value in data_column:
            if FileProcessor._is_blank_value(value):
                original_data.append(None)
                blank_count += 1
            else:
                try:
                    numeric_value = float(value)
                    original_data.append(numeric_value)
                    clean_data.append(numeric_value)
                    
                    # 分析小数位数
                    str_value = str(numeric_value)
                    if '.' in str_value:
                        decimal_part = str_value.split('.')[1].rstrip('0')
                        decimal_places = len(decimal_part)
                    else:
                        decimal_places = 0
                    
                    # 更新最大小数位数
                    max_decimal_places = max(max_decimal_places, decimal_places)
                    
                    # 统计小数位数
                    decimal_info['decimal_places_count'][decimal_places] = \
                        decimal_info['decimal_places_count'].get(decimal_places, 0) + 1
                    decimal_info['max_decimal_places'] = max_decimal_places
                    
                except (ValueError, TypeError):
                    original_data.append(None)
                    blank_count += 1
        
        # 确定检测到的小数位数 - 使用最大小数位数
        decimal_info['detected_decimal_places'] = max_decimal_places
        
        # 显示处理结果
        if blank_count > 0:
            st.warning(f"检测到 {blank_count} 个空白或无效数据，已自动过滤")
        
        return np.array(clean_data), original_data, blank_count, decimal_info

# =============================================
# 初始化会话状态
# =============================================

def initialize_session_state():
    """初始化所有必要的会话状态变量"""
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6"
    
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
        
    if 'validation_report' not in st.session_state:
        st.session_state.validation_report = []
        
    if 'validation_passed' not in st.session_state:
        st.session_state.validation_passed = False
        
    if 'decimal_info' not in st.session_state:
        st.session_state.decimal_info = {}
    
    # 两列数据相关的会话状态
    if 'two_column_data' not in st.session_state:
        st.session_state.two_column_data = ""
    
    if 'two_column_processed' not in st.session_state:
        st.session_state.two_column_processed = False
    
    if 'label_data_pairs' not in st.session_state:
        st.session_state.label_data_pairs = []
    
    if 'two_column_validation_report' not in st.session_state:
        st.session_state.two_column_validation_report = []
        
    if 'valid_pairs' not in st.session_state:
        st.session_state.valid_pairs = []
        
    if 'original_labels' not in st.session_state:
        st.session_state.original_labels = []
    
    # 文件上传相关的会话状态
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
        
    if 'file_decimal_info' not in st.session_state:
        st.session_state.file_decimal_info = {}
        
    if 'two_column_decimal_info' not in st.session_state:
        st.session_state.two_column_decimal_info = {}
    
    # 计算方案相关的会话状态
    if 'calculation_scheme' not in st.session_state:
        st.session_state.calculation_scheme = "严格计算方案"

# =============================================
# 缺失的函数定义
# =============================================

def analyze_data():
    """分析手动输入的数据"""
    try:
        # 获取当前输入的数据
        data_string = st.session_state.manual_data
        
        # 检查数据是否为空
        if not data_string or data_string.strip() == "":
            st.error("❌ 请输入数据")
            return
        
        # 从会话状态获取计算方案
        calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
        
        # 进行数据验证
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
            
            # 保存到历史记录（只保存不同的数据）
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
        # 移除当前状态
        st.session_state.data_history.pop()
        # 恢复上一个状态（如果有）
        if len(st.session_state.data_history) > 0:
            previous_data = st.session_state.data_history[-1]
            st.session_state.manual_data = previous_data
            # 重新分析数据
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
        # 获取输入数据
        two_column_input = st.session_state.two_column_data
        
        # 从会话状态获取计算方案
        calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
        
        # 验证两列数据
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
        
        # 提取数据
        data = np.array([value for _, value in valid_pairs])
        labels = [label for label, _ in valid_pairs]
        
        # 存储到会话状态
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
# 主程序开始
# =============================================

# 设置页面
st.set_page_config(
    page_title="统计宝 | 稳健统计分析工具 ",
    page_icon="stataid.png",
    layout="wide"
)

initialize_session_state()

# 设置页面配置
# 添加CSS，恢复原有设定并添加底部空白
st.markdown("""
<style>
/* 为Streamlit Cloud顶部UI元素保留空间 */
.stApp {
    margin-top: 0 !important;
    padding-top: 30px !important; /* 为顶部按钮/菜单留出空间 */
}

/* 确保内容区域有适当间距 */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* 主容器 - 单行显示 */
.single-line-container {
    display: flex !important;
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 10px 0 !important; /* 添加少量内边距 */
}

/* 图标容器 */
.icon-wrapper {
    flex: 0 0 auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    margin-right: 30px !important;
}

/* 文字容器 - 保持固定上边距 */
.text-wrapper {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    padding-top: 60px !important;
}

/* 响应式图标大小调整 */
@media (max-width: 1200px) {
    .single-line-container img {
        width: 280px !important;
    }
}

@media (max-width: 992px) {
    .single-line-container img {
        width: 220px !important;
    }
    .text-wrapper {
        padding-top: 35px !important;
    }
}

@media (max-width: 768px) {
    .single-line-container img {
        width: 180px !important;
    }
    .text-wrapper {
        padding-top: 30px !important;
    }
}

@media (max-width: 576px) {
    .single-line-container img {
        width: 150px !important;
    }
    .text-wrapper {
        padding-top: 25px !important;
    }
}

@media (max-width: 480px) {
    .single-line-container img {
        width: 120px !important;
    }
    .text-wrapper {
        padding-top: 20px !important;
    }
}

/* 强制在小屏幕上保持单行 */
@media (max-width: 768px) {
    .single-line-container {
        flex-wrap: nowrap !important;
    }
}

/* 确保Streamlit顶部工具栏不被遮挡 */
header[data-testid="stHeader"] {
    z-index: 1000 !important;
}

/* 如果顶部菜单仍然被遮挡，可以增加此值 */
@media (max-width: 768px) {
    .stApp {
        padding-top: 30px !important; /* 在手机上可能需要更多空间 */
    }
}

/* 统计分析结果 - 统一的2行4列布局 */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: auto auto;
    gap: 10px;
    margin-bottom: 20px;
}

.stats-metric {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e9ecef;
}

/* 为底部添加空白，确保用户反馈模块完整显示 */
.main-footer {
    margin-top: 50px;
    padding: 20px 0;
    background-color: #f8f9fa;
    border-top: 1px solid #e9ecef;
}

/* 添加页面底部空白 */
.page-bottom-padding {
    height: 100px;
    clear: both;
}

/* 确保统计量显示对齐 */
.aligned-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
}

.aligned-metric-item {
    flex: 1 0 calc(25% - 10px);
    min-width: 200px;
}
</style>
""", unsafe_allow_html=True)

# 加载软件图标
icon = Image.open("stataid_cut edge.png")

# 使用容器包裹
with st.container():
    # 使用自定义CSS类 - 恢复原有设定
    st.markdown('<div class="single-line-container">', unsafe_allow_html=True)
    
    # 创建两列
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # 图标容器 - 使用负边距向左对齐
        st.markdown('<div class="icon-wrapper" style="margin-left: -30px;">', unsafe_allow_html=True)
        st.image(icon, width=360)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # 文字容器 - 恢复原有设定
        st.markdown('<div class="text-wrapper">', unsafe_allow_html=True)
        st.markdown("### **统计宝**")
        st.markdown("提供多种稳健统计分析方法，用于处理包含异常值的数据集。")
        # 删除杂乱的副标题列表，恢复简洁版本
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# 优化侧边栏布局
# =============================================

# 在侧边栏 - 参数设置和方法选择部分
st.sidebar.header("⚙️ 分析设置")

# 方法选择
method = st.sidebar.selectbox(
    "选择统计方法:",
    ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法", "Z比分计算模块"],
    help="选择适合数据特征的稳健统计方法"
)

# 动态参数显示
if method == "迭代稳健统计法":
    st.sidebar.subheader("迭代法参数")
    k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
    max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)

# 注意：Z比分计算模块的参数输入已移到主界面，此处不再显示

# 计算方案选择
st.sidebar.subheader("计算方案")
calculation_scheme = st.sidebar.radio(
    "选择计算方案:",
    ["严格计算方案", "规范展示方案"],
    help="""
    严格计算方案：使用完整精度的计算结果，确保计算准确性
    规范展示方案：稳健平均值与原始数据小数位数一致，结果更规范但可能引入微小误差
    """,
    key="calculation_scheme_selector"  # 添加key
)

# 将计算方案保存到会话状态
st.session_state.calculation_scheme = calculation_scheme

# 高级选项
st.sidebar.subheader("高级选项")
show_scheme_comparison = st.sidebar.checkbox(
    "显示方案比较", 
    help="同时显示两种计算方案的结果对比",
    value=False
)

# 方法说明
st.sidebar.markdown("---")
st.sidebar.header("📚 方法说明")

method_descriptions = {
    "迭代稳健统计法": "通过迭代过程逐步修正异常值影响，收敛后得到稳健的统计估计。",
    "四分位稳健统计法": "以数据排序为基础，使用数据集中段50%的数据，崩溃点为25%，具有易于计算、操作简单的特点。",
    "Q/Hampel法": "结合Q方法计算的稳健标准差和Hampel方法计算的稳健平均值，具有较好的抗异常值干扰能力。",
    "Z比分计算模块": "使用用户提供的稳健统计量计算Z比分：Z比分 = (测试数据-稳健平均值)/稳健标准差"
}

st.sidebar.info(method_descriptions[method])

# 特别为Z比分计算模块添加说明
if method == "Z比分计算模块":
    st.sidebar.info("💡 **注意**: Z比分计算参数请在主界面输入")

# 数据输入方式选择 - 修复版
st.markdown("### **👉数据输入方式**")
input_method = st.radio("", 
                       ["手动输入", "带编号数据输入", "文件上传", "示例数据"],
                       horizontal=True,
                       index=0,  # 设置默认选中第一个选项
                       label_visibility="collapsed")

# 手动输入：
if input_method == "手动输入":
    st.subheader("📝 手动输入数据")
    
    # 简化的输入界面
    manual_input = st.text_area(
        "请输入数据（每行一个数值或用逗号分隔）:",
        value=st.session_state.manual_data,
        height=150,
        key=f"manual_input_{st.session_state.reset_counter}",
        help="支持空白数据自动忽略"
    )
    
    # 更新会话状态
    st.session_state.manual_data = manual_input
    
    # 操作按钮
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("分析数据", use_container_width=True, type="primary"):
            analyze_data()
    
    with col2:
        if st.button("一键清除", use_container_width=True, type="secondary"):
            clear_data()
    
    with col3:
        undo_disabled = len(st.session_state.data_history) <= 1  # 只有当前数据时禁用
        undo_label = f"↶ 撤销 ({len(st.session_state.data_history)-1}次可用)" if len(st.session_state.data_history) > 1 else "↶ 撤销 (无历史)"
        if st.button(undo_label, use_container_width=True, disabled=undo_disabled):
            undo_data()
    
    # 显示验证结果 - 修改：仅显示小数位数保留规则的介绍，默认为收起状态
    if st.session_state.validation_passed:
        # 获取小数位数信息
        decimal_places = st.session_state.decimal_info.get('detected_decimal_places', 0)
        max_decimal_places = st.session_state.decimal_info.get('max_decimal_places', 0)
        consistent_decimals = st.session_state.decimal_info.get('consistent_decimals', True)
        
        with st.expander("📋 小数位数保留规则说明", expanded=False):
            st.info(f"""
            **小数位数处理规则：**
            1. 检测到数据最大小数位数: {max_decimal_places}位
            2. 数据小数位数一致性: {'一致' if consistent_decimals else '不一致'}
            3. 使用的小数位数: {decimal_places}位
            """)
    
    # 如果已经处理了数据，则设置data变量
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        if st.session_state.blank_count > 0:
            st.warning(f"⚠️ 检测到 {st.session_state.blank_count} 个空白数据点，已忽略")

elif input_method == "带编号数据输入":
    st.subheader("🏷️ 带编号数据输入")
    
    # 两列数据输入说明
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
    
    # 两列数据输入框
    two_column_input = st.text_area(
        "请输入标签和数值数据（每行一个数据对，用逗号分隔）:",
        value=st.session_state.two_column_data,
        height=200,
        key="two_column_input",
        help="格式：标签, 数值"
    )
    
    # 更新会话状态
    st.session_state.two_column_data = two_column_input
    
    # 分析按钮
    if st.button("分析两列数据", type="primary", use_container_width=True):
        analyze_two_column_data()
    
    # 清除按钮
    if st.button("清除两列数据", type="secondary", use_container_width=True):
        clear_two_column_data()
    
    # 如果已经处理了两列数据，则设置data变量
    if st.session_state.two_column_processed and st.session_state.processed_data is not None:
        data = st.session_state.processed_data

elif input_method == "文件上传":
    st.subheader("📁 上传数据文件")
    
    # 扩展支持的文件类型
    uploaded_file = st.file_uploader(
        "选择数据文件 (支持 CSV、TXT、Excel、JSON)", 
        type=['csv', 'txt', 'xlsx', 'xls', 'json'],
        help="支持多种文件格式：CSV、文本文件、Excel工作簿、JSON数据文件。空白数据会自动识别并忽略。"
    )
    
    if uploaded_file is not None:
        try:
            # 显示文件信息
            file_size = uploaded_file.size / 1024  # KB
            st.write(f"📄 **文件信息**: {uploaded_file.name} ({file_size:.1f} KB)")
            
            # 自动检测文件格式
            file_format = FileProcessor.detect_file_format(uploaded_file)
            st.write(f"🔍 **检测到的格式**: {file_format.upper()}")
            
            processed_data = None
            original_data = []
            blank_count = 0
            
            # 根据文件格式调用相应的处理方法
            if file_format == 'excel':
                df, sheet_name, all_sheets = FileProcessor.process_excel_file(uploaded_file)
            elif file_format == 'csv':
                df, sheet_name, all_sheets = FileProcessor.process_csv_file(uploaded_file)
            elif file_format == 'json':
                df, sheet_name, all_sheets = FileProcessor.process_json_file(uploaded_file)
            else:  # txt
                df, sheet_name, all_sheets = FileProcessor.process_txt_file(uploaded_file)
            
            # 统一处理所有格式的DataFrame
            if df is not None:
                # 从DataFrame提取数据
                clean_data, original_data, blank_count, decimal_info = FileProcessor.extract_data_from_dataframe(df, sheet_name)
                
                if clean_data is not None and len(clean_data) > 0:
                    processed_data = clean_data
                    
                    # 构建验证报告
                    validation_report = [
                        "✅ 文件格式验证通过",
                        f"✅ 成功从 '{sheet_name}' 提取数据",
                        f"📊 总数据点数: {len(original_data)}",
                        f"📈 有效数据数: {len(clean_data)}",
                        f"⚠️ 空白数据数: {blank_count}" if blank_count > 0 else "✅ 未发现空白数据",
                        f"📏 检测到的小数位数: {decimal_info['detected_decimal_places']}位"
                    ]
                    
                    # 获取计算方案
                    calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
                    
                    # 计算方案兼容性验证
                    scheme_messages = DataValidator.validate_calculation_scheme_compatibility(
                        clean_data, calculation_scheme, decimal_info
                    )
                    validation_report.extend(scheme_messages)
                    
                    # 推荐计算方案
                    recommended_scheme, recommendation_reason = DataValidator.get_recommended_scheme(decimal_info)
                    validation_report.append(f"💡 推荐计算方案: {recommended_scheme} - {recommendation_reason}")
                    
                    # 设置正确的会话状态，确保使用文件数据
                    st.session_state.file_processed_data = processed_data
                    st.session_state.file_original_data = original_data
                    st.session_state.file_blank_count = blank_count
                    st.session_state.file_validation_report = validation_report
                    st.session_state.file_validation_passed = True
                    st.session_state.file_decimal_info = decimal_info
                    
                    # 同时设置通用状态，确保后续分析使用文件数据
                    st.session_state.processed_data = processed_data
                    st.session_state.original_data = original_data
                    st.session_state.blank_count = blank_count
                    st.session_state.decimal_info = decimal_info
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ 文件验证通过！成功加载 {len(processed_data)} 个有效数据点")
                    if blank_count > 0:
                        st.warning(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
                    
                    # 显示小数位数保留规则说明
                    decimal_places = decimal_info.get('detected_decimal_places', 0)
                    max_decimal_places = decimal_info.get('max_decimal_places', 0)
                    consistent_decimals = decimal_info.get('consistent_decimals', True)
                    
                    with st.expander("📋 小数位数保留规则说明", expanded=False):
                        st.info(f"""
                        **小数位数处理规则：**
                        1. 检测到数据最大小数位数: {max_decimal_places}位
                        2. 数据小数位数一致性: {'一致' if consistent_decimals else '不一致'}
                        3. 使用的小数位数: {decimal_places}位
                        """)
                    
                    st.write("**前10个有效数据:**", processed_data[:10])
                    
                    # 设置数据变量，以便后续分析
                    data = processed_data
                    
                else:
                    st.error("❌ 无法从文件中提取有效数据")
            
            else:
                st.error("❌ 文件数据验证失败或没有有效数据")
                if hasattr(st.session_state, 'file_validation_report') and st.session_state.file_validation_report:
                    with st.expander("📋 查看验证详情", expanded=True):
                        for line in st.session_state.file_validation_report:
                            if line.startswith("❌"):
                                st.error(line)
                            else:
                                st.write(line)
                else:
                    st.error("无法从文件中提取有效数据，请检查文件格式和内容")
            
        except Exception as e:
            st.error(f"❌ 文件处理错误: {str(e)}")
            st.info("💡 请确保文件格式正确且包含有效的数值数据")
    
    # 如果已经处理了文件数据，则设置data变量
    if st.session_state.file_validation_passed and st.session_state.file_processed_data is not None:
        data = st.session_state.file_processed_data

else:  # 示例数据
    st.subheader("🎯 示例数据分析")
    example_data = np.array([
        54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
        54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
        54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
        54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
        54.5, 55.9, 53.2, 54.6
    ])
    
    # 验证示例数据
    calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
    is_valid, original_data, clean_data, blank_count, validation_report, decimal_info = \
        DataValidator.comprehensive_validation(
            "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6",
            calculation_scheme
        )
    
    # 确认使用示例数据
    if st.button("使用示例数据进行分析", type="primary"):
        data = example_data
        st.session_state.processed_data = example_data
        st.session_state.original_data = example_data.tolist()
        st.session_state.data_loaded = True
        st.session_state.blank_count = 0
        st.session_state.decimal_info = decimal_info
        
        st.success(f"✅ 示例数据已加载，包含 {len(example_data)} 个测量值")
        st.rerun()
    
    if is_valid:
        st.success("✅ 示例数据验证通过")
    
    # 显示小数位数保留规则说明
    decimal_places = decimal_info.get('detected_decimal_places', 0)
    max_decimal_places = decimal_info.get('max_decimal_places', 0)
    consistent_decimals = decimal_info.get('consistent_decimals', True)
    
    with st.expander("📋 小数位数保留规则说明", expanded=False):
        st.info(f"""
        **小数位数处理规则：**
        1. 检测到数据最大小数位数: {max_decimal_places}位
        2. 数据小数位数一致性: {'一致' if consistent_decimals else '不一致'}
        3. 使用的小数位数: {decimal_places}位
        """)
    
    # 设置数据变量，以便后续分析
    data = example_data
    
    # 为示例数据设置必要的会话状态变量，以便导出模块正常工作
    st.session_state.original_data = example_data.tolist()  # 转换为列表格式
    st.session_state.blank_count = 0  # 示例数据没有空白
    # 保存小数位数信息
    st.session_state.decimal_info = decimal_info

# =============================================
# Z比分计算模块的数据输入（只在主界面显示）
# =============================================

if method == "Z比分计算模块":
    st.markdown("---")
    st.subheader("🔢 Z比分计算参数")
    
    # 在主界面显示稳健统计量输入
    col1, col2 = st.columns(2)
    
    with col1:
        robust_mean_input = st.text_input(
            "稳健平均值:",
            value="54.4",
            help="请输入稳健平均值",
            key="robust_mean_input"
        )
    
    with col2:
        robust_std_input = st.text_input(
            "稳健标准差:",
            value="0.3",
            help="请输入稳健标准差",
            key="robust_std_input"
        )
    
    # 添加说明
    st.info("""
    **Z比分计算公式：**
    ```
    Z比分 = (测试数据 - 稳健平均值) / 稳健标准差
    ```
    请确保输入的稳健统计量准确无误。
    """)

# =============================================
# 根据输入方式重新设置data变量
# =============================================

if input_method == "手动输入":
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        data = st.session_state.processed_data
    else:
        data = None
elif input_method == "带编号数据输入":
    if st.session_state.two_column_processed and st.session_state.processed_data is not None:
        data = st.session_state.processed_data
    else:
        data = None
elif input_method == "文件上传":
    if st.session_state.file_validation_passed and st.session_state.file_processed_data is not None:
        data = st.session_state.file_processed_data
    else:
        data = None
else:  # 示例数据
    data = example_data  # 确保example_data已经被定义

# =============================================
# 统计方法实现 - 修改版本
# =============================================

def detect_decimal_places(data):
    """检测数据的小数位数 - 返回最大小数位数"""
    if data is None or (hasattr(data, '__len__') and len(data) == 0):
        return 0
    
    max_decimal_places = 0
    
    try:
        # 确保数据是可迭代的
        data_array = np.asarray(data)
        for value in data_array:
            if isinstance(value, (int, float)) and not np.isnan(value):
                # 将数值转换为字符串
                str_value = str(value)
                
                # 处理科学计数法
                if 'e' in str_value.lower():
                    # 如果是科学计数法，转换为普通小数表示
                    str_value = format(value, '.15f')
                
                # 分割整数和小数部分
                if '.' in str_value:
                    decimal_part = str_value.split('.')[1]
                    # 去除末尾的零（如果有的话）
                    decimal_part = decimal_part.rstrip('0')
                    current_decimal_places = len(decimal_part)
                    max_decimal_places = max(max_decimal_places, current_decimal_places)
        
        return max_decimal_places
        
    except Exception as e:
        # 如果出现任何错误，返回默认值2
        return 2

def iterative_robust_algorithm(data, max_iterations=50, k=1.5, scheme="strict"):
    """迭代稳健统计法 - 修复 decimal_places 问题"""
    # 在函数开头确保 decimal_places 有默认值
    decimal_places = 0
    
    # 确保数据有效
    if data is None or len(data) == 0:
        # 返回空的但完整的结果结构
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
            'decimal_places': 0,  # 确保有值
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
    
    # 检测数据的小数位数 - 确保在任何情况下都有值
    try:
        decimal_places = detect_decimal_places(data)
    except:
        decimal_places = 0  # 如果检测失败，使用默认值
    
    # 根据选择的方案进行格式化
    if scheme == "presentation":
        # 规范展示方案：使用四舍五入后的均值和标准差
        formatted_X_star = round(X_star, decimal_places)
        formatted_S_star = round(S_star, 3)
        
        # 使用格式化后的值计算Z比分
        Z_scores_high_precision = (data - formatted_X_star) / formatted_S_star
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_X_star})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
        
        robust_mean = formatted_X_star
        robust_std = formatted_S_star
        
    else:
        # 严格计算方案：使用原始计算值
        formatted_X_star = X_star
        formatted_S_star = S_star
        
        # 使用原始计算值计算Z比分
        Z_scores_high_precision = (data - X_star) / S_star
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
        
        robust_mean = X_star
        robust_std = S_star
    
    # 计算界限
    final_delta = k * robust_std
    lower_limit = robust_mean - final_delta
    upper_limit = robust_mean + final_delta
    
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    
    # 彻底的类型安全处理
    outliers_list = []
    clean_data_list = []
    
    # 确保数据是可迭代的
    if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
        data_iter = data
    else:
        data_iter = [data]
    
    # 确保掩码是可迭代的
    if hasattr(outliers_mask, '__iter__') and not isinstance(outliers_mask, (str, dict)):
        mask_iter = outliers_mask
    else:
        mask_iter = [outliers_mask]
    
    # 安全地分离异常值和正常数据
    for i, value in enumerate(data_iter):
        if i < len(mask_iter) and mask_iter[i]:
            try:
                outliers_list.append(float(value))
            except (ValueError, TypeError):
                continue
        else:
            try:
                clean_data_list.append(float(value))
            except (ValueError, TypeError):
                continue
    
    # 确保Z_scores是安全的Python类型
    safe_z_scores_high_precision = Z_scores_high_precision.tolist() if hasattr(Z_scores_high_precision, 'tolist') else [float(z) for z in Z_scores_high_precision]
    safe_z_scores_rounded = Z_scores_rounded.tolist() if hasattr(Z_scores_rounded, 'tolist') else [float(z) for z in Z_scores_rounded]
    
    # 格式化Z比分为两位小数用于显示
    formatted_Z_scores = format_z_scores(safe_z_scores_rounded)
    
    # 为每个数据点生成分类
    z_score_classifications = [classify_z_score(z) for z in safe_z_scores_rounded]
    
    # 确保返回标准Python类型
    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores_high_precision': safe_z_scores_high_precision,  # 高精度Z比分
        'Z_scores_rounded': safe_z_scores_rounded,  # 保留两位小数的Z比分
        'formatted_Z_scores': formatted_Z_scores,  # 格式化显示的Z比分
        'z_score_classifications': z_score_classifications,  # Z比分分类
        'iterations': iteration,
        'converged': converged,
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'history': history,
        'method_name': '迭代稳健统计法',
        'decimal_places': decimal_places,  # 确保这个字段总是有值
        'calculation_scheme': scheme,
        'formatting_note': formatting_note,
        'original_mean': float(X_star) if not np.isnan(X_star) else 0.0,  # 始终保存原始计算值
        'original_std': float(S_star) if not np.isnan(S_star) else 0.0    # 始终保存原始计算值
    }

def quartile_robust_algorithm(data, scheme="strict"):
    """四分位稳健统计法 - 支持两种计算方案和新的Z比分分类"""
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
    # 彻底的类型安全处理
    outliers_list = []
    clean_data_list = []
    
    # 确保数据是可迭代的
    if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
        data_iter = data
    else:
        data_iter = [data]
    
    # 确保掩码是可迭代的
    if hasattr(outliers_mask, '__iter__') and not isinstance(outliers_mask, (str, dict)):
        mask_iter = outliers_mask
    else:
        mask_iter = [outliers_mask]
    
    # 安全地分离异常值和正常数据
    for i, value in enumerate(data_iter):
        if i < len(mask_iter) and mask_iter[i]:
            try:
                outliers_list.append(float(value))
            except (ValueError, TypeError):
                continue
        else:
            try:
                clean_data_list.append(float(value))
            except (ValueError, TypeError):
                continue
    
    # 检测数据的小数位数 - 与迭代法保持一致
    decimal_places = detect_decimal_places(data)
    
    # 根据选择的方案进行格式化
    if scheme == "presentation":
        # 规范展示方案：使用四舍五入后的均值和标准差
        formatted_median = round(median, decimal_places)
        formatted_niqr = round(niqr, 3)
        
        # 使用格式化后的值计算Z比分
        Z_scores_high_precision = (data - formatted_median) / formatted_niqr
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_median})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
        
        robust_mean = formatted_median
        robust_std = formatted_niqr
        
        # 在规范展示方案中，其他统计量也按原始数据小数位数格式化
        formatted_q1 = round(q1, decimal_places)
        formatted_q3 = round(q3, decimal_places)
        formatted_iqr = round(iqr, decimal_places)
        formatted_niqr_display = round(niqr, 3)  # niqr已经格式化过了
        
    else:
        # 严格计算方案：使用原始计算值
        formatted_median = median
        formatted_niqr = niqr
        
        # 使用原始计算值计算Z比分
        Z_scores_high_precision = (data - median) / niqr
        Z_scores_rounded = np.round(Z_scores_high_precision, 2)
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
        
        robust_mean = median
        robust_std = niqr
        
        # 在严格计算方案中，显示6位小数
        formatted_q1 = q1
        formatted_q3 = q3
        formatted_iqr = iqr
        formatted_niqr_display = niqr
    
    # 确保Z_scores是安全的Python类型
    safe_z_scores_high_precision = Z_scores_high_precision.tolist() if hasattr(Z_scores_high_precision, 'tolist') else [float(z) for z in Z_scores_high_precision]
    safe_z_scores_rounded = Z_scores_rounded.tolist() if hasattr(Z_scores_rounded, 'tolist') else [float(z) for z in Z_scores_rounded]

    # 格式化Z比分为两位小数用于显示
    formatted_Z_scores = format_z_scores(safe_z_scores_rounded)
    
    # 为每个数据点生成分类
    z_score_classifications = [classify_z_score(z) for z in safe_z_scores_rounded]

    # 确保返回标准Python类型
    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores_high_precision': safe_z_scores_high_precision,  # 高精度Z比分
        'Z_scores_rounded': safe_z_scores_rounded,  # 保留两位小数的Z比分
        'formatted_Z_scores': formatted_Z_scores,  # 格式化显示的Z比分
        'z_score_classifications': z_score_classifications,  # Z比分分类
        'q1': float(formatted_q1) if not np.isnan(formatted_q1) else 0.0,
        'q3': float(formatted_q3) if not np.isnan(formatted_q3) else 0.0,
        'iqr': float(formatted_iqr) if not np.isnan(formatted_iqr) else 0.0,
        'niqr': float(formatted_niqr_display) if not np.isnan(formatted_niqr_display) else 0.0,
        'method_name': '四分位稳健统计法',
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'formatting_note': formatting_note,
        'calculation_scheme': scheme,
        'decimal_places': decimal_places,  # 添加小数位数信息，与迭代法保持一致
        'original_mean': float(median) if not np.isnan(median) else 0.0,  # 保存原始计算值
        'original_std': float(niqr) if not np.isnan(niqr) else 0.0  # 保存原始计算值
    }

# =============================================
# Z比分计算模块 - 完整实现
# =============================================

def z_score_calculation_algorithm(data, robust_mean, robust_std, scheme="strict"):
    """
    Z比分计算方法 - 使用用户提供的稳健统计量
    完整实现严格计算方案和规范展示方案
    """
    try:
        # 确保数据是numpy数组
        data_array = np.asarray(data, dtype=float)
        
        # 验证稳健统计量
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
        
        # 检测数据的小数位数
        decimal_places = detect_decimal_places(data_array)
        
        # 根据选择的方案进行格式化
        if scheme == "presentation":
            # 规范展示方案
            formatted_robust_mean = round(robust_mean_val, decimal_places)
            formatted_robust_std = round(robust_std_val, 3)
            
            # 使用格式化后的值计算Z比分（计算过程不四舍五入）
            Z_scores_high_precision = (data_array - formatted_robust_mean) / formatted_robust_std
            Z_scores_rounded = np.round(Z_scores_high_precision, 2)
            
            formatting_note = f"使用规范展示方案：稳健平均值({formatted_robust_mean})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
            
            robust_mean_display = formatted_robust_mean
            robust_std_display = formatted_robust_std
            
        else:
            # 严格计算方案
            formatted_robust_mean = robust_mean_val
            formatted_robust_std = robust_std_val
            
            # 使用原始计算值计算Z比分
            Z_scores_high_precision = (data_array - robust_mean_val) / robust_std_val
            Z_scores_rounded = np.round(Z_scores_high_precision, 2)
            
            formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
            
            robust_mean_display = robust_mean_val
            robust_std_display = robust_std_val
        
        # 计算正常值范围
        lower_limit = robust_mean_display - 3 * robust_std_display
        upper_limit = robust_mean_display + 3 * robust_std_display
        
        # 识别离群值
        outliers_mask = (data_array < lower_limit) | (data_array > upper_limit)
        
        # 分离正常数据和离群值
        outliers_list = []
        clean_data_list = []
        
        for i, value in enumerate(data_array):
            if outliers_mask[i]:
                outliers_list.append(float(value))
            else:
                clean_data_list.append(float(value))
        
        # 确保Z_scores是安全的Python类型
        safe_z_scores_high_precision = Z_scores_high_precision.tolist() if hasattr(Z_scores_high_precision, 'tolist') else [float(z) for z in Z_scores_high_precision]
        safe_z_scores_rounded = Z_scores_rounded.tolist() if hasattr(Z_scores_rounded, 'tolist') else [float(z) for z in Z_scores_rounded]
        
        # 格式化Z比分为两位小数用于显示
        formatted_Z_scores = format_z_scores(safe_z_scores_rounded)
        
        # 为每个数据点生成分类
        z_score_classifications = [classify_z_score(z) for z in safe_z_scores_rounded]
        
        return {
            'robust_mean': float(robust_mean_display),
            'robust_std': float(robust_std_display),
            'clean_data': clean_data_list,
            'outliers': outliers_list,
            'Z_scores_high_precision': safe_z_scores_high_precision,  # 高精度Z比分
            'Z_scores_rounded': safe_z_scores_rounded,  # 保留两位小数的Z比分
            'formatted_Z_scores': formatted_Z_scores,  # 格式化显示的Z比分
            'z_score_classifications': z_score_classifications,  # Z比分分类
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
# Q/Hampel法实现 - 修正稳健平均值计算
# =============================================

def perform_Q_estimate_corrected(x_data):
    '''
    修正后的Q方法实现，确保G1值计算正确
    '''
    if isinstance(x_data, pd.Series):
        x_values = x_data.astype('float').values
    else:
        x_values = np.asarray(x_data, dtype=float)
    
    x_values = x_values[~np.isnan(x_values)]
    p = len(x_values)
    
    print(f"数据点数: {p}")
    print(f"数据范围: [{np.min(x_values)}, {np.max(x_values)}]")
    
    # 计算所有成对绝对差值
    d_arr = []
    for i in range(p-1):
        for j in range(i+1, p):
            d = np.abs(x_values[i] - x_values[j])
            d_arr.append(d)
    
    d_arr = np.array(d_arr)
    print(f"成对差值数量: {len(d_arr)}")
    print(f"差值范围: [{np.min(d_arr)}, {np.max(d_arr)}]")
    
    # 计算H(0)
    H1_0 = np.mean(d_arr <= 0)
    print(f"H(0) = {H1_0}")
    
    # 找到不连续点
    discontinuity_points = np.unique(d_arr)
    print(f"不连续点数量: {len(discontinuity_points)}")
    
    # 计算每个不连续点的H1值
    H1_values = [np.mean(d_arr <= point) for point in discontinuity_points]
    
    # 修正G1值计算 - 严格按照ISO标准
    G1_values = []
    G1_points = []
    
    # 首先添加G1(0)=0
    G1_points.append(0.0)
    G1_values.append(0.0)
    
    # 计算每个间断点的G1值
    for i, point in enumerate(discontinuity_points):
        if i == 0 and point > 0:
            # i=1, x_1>0: G1(x_1) = 0.5 * H1(x_1)
            G1_val = 0.5 * H1_values[i]
            G1_points.append(point)
            G1_values.append(G1_val)
        elif i >= 1:
            # i≥2: G1(x_i) = 0.5 * [H1(x_i) + H1(x_{i-1})]
            G1_val = 0.5 * (H1_values[i] + H1_values[i-1])
            G1_points.append(point)
            G1_values.append(G1_val)
    
    # 确保G1_points是排序的
    G1_points = np.array(G1_points)
    G1_values = np.array(G1_values)
    
    # 线性插值 + 边界外推
    if len(G1_points) > 1:
        # 创建密集的插值点
        n_interp = 10000
        x_interp = np.linspace(G1_points[0], G1_points[-1], n_interp)
        
        # 使用线性插值，允许外推
        G1_interp_func = interpolate.interp1d(
            G1_points, G1_values, 
            kind='linear', 
            bounds_error=False,
            fill_value=(G1_values[0], G1_values[-1]),
            assume_sorted=True
        )
        G1_interp = G1_interp_func(x_interp)
        print("使用线性插值 + 边界外推")
    else:
        x_interp = G1_points
        G1_interp = G1_values
        print("只有一个点，无需插值")
    
    # 计算分子
    target_G1 = 0.25 + 0.75 * H1_0
    print(f"目标G1值: {target_G1}")
    
    # 寻找G1反函数 - 精确方法
    # 找到跨越目标值的区间
    idx_above = np.where(G1_interp >= target_G1)[0]
    idx_below = np.where(G1_interp <= target_G1)[0]
    
    if len(idx_above) > 0 and len(idx_below) > 0:
        idx_high = idx_above[0]
        idx_low = idx_below[-1]
        
        if idx_high == idx_low:
            numerator = x_interp[idx_high]
        else:
            # 线性插值
            x1, x2 = x_interp[idx_low], x_interp[idx_high]
            y1, y2 = G1_interp[idx_low], G1_interp[idx_high]
            
            if abs(y2 - y1) > 1e-12:
                numerator = x1 + (x2 - x1) * (target_G1 - y1) / (y2 - y1)
            else:
                numerator = (x1 + x2) / 2
                
        print(f"找到跨越区间: [{x_interp[idx_low]:.6f}, {x_interp[idx_high]:.6f}]")
        print(f"对应G1值: [{G1_interp[idx_low]:.6f}, {G1_interp[idx_high]:.6f}]")
    else:
        # 边界情况 - 使用外推
        if len(idx_above) == 0:  # 所有点都小于目标值
            numerator = x_interp[-1]
            print("目标值超出右边界，使用最大值")
        else:  # 所有点都大于目标值
            numerator = x_interp[0]
            print("目标值超出左边界，使用最小值")
    
    print(f"分子: {numerator:.10f}")
    
    # 计算分母
    target_phi = 0.625 + 0.375 * H1_0
    target_phi = np.clip(target_phi, 0.001, 0.999)
    denominator = np.sqrt(2) * norm.ppf(target_phi)
    
    print(f"目标Φ值: {target_phi:.10f}")
    print(f"Φ^(-1)(目标Φ值): {norm.ppf(target_phi):.10f}")
    print(f"分母: {denominator:.10f}")
    
    Q = numerator / denominator if denominator > 1e-12 else 0.0
    print(f"Q估计: {Q:.10f}")
    
    return Q

def hampel_robust_mean(data, max_iterations=50, tol=1e-8):
    """
    修正的Hampel稳健平均值计算方法
    使用迭代重加权最小二乘法
    """
    if len(data) == 0:
        return 0.0
    
    # 初始估计使用中位数
    x = np.median(data)
    
    for iteration in range(max_iterations):
        # 计算残差
        residuals = data - x
        
        # 计算MAD（中位数绝对偏差）
        mad = np.median(np.abs(residuals))
        
        # 如果MAD为0，说明所有数据相同，直接返回
        if mad < 1e-12:
            return float(x)
        
        # 标准化残差
        u = residuals / (1.4826 * mad)
        
        # Hampel权重函数
        weights = np.ones_like(data)
        abs_u = np.abs(u)
        
        # 三部分权重函数
        # 1. |u| ≤ 1.5: 权重 = 1
        # 2. 1.5 < |u| ≤ 3.0: 权重 = 1.5/|u|
        mask1 = (abs_u > 1.5) & (abs_u <= 3.0)
        weights[mask1] = 1.5 / abs_u[mask1]
        
        # 3. 3.0 < |u| ≤ 4.5: 权重 = (4.5 - |u|)/1.5 * 0.5
        mask2 = (abs_u > 3.0) & (abs_u <= 4.5)
        weights[mask2] = (4.5 - abs_u[mask2]) / 1.5 * 0.5
        
        # 4. |u| > 4.5: 权重 = 0
        mask3 = abs_u > 4.5
        weights[mask3] = 0
        
        # 计算新的估计值
        total_weight = np.sum(weights)
        if total_weight < 1e-12:
            # 如果所有权重都为0，返回当前估计
            return float(x)
        
        x_new = np.sum(weights * data) / total_weight
        
        # 检查收敛
        if abs(x_new - x) < tol:
            return float(x_new)
        
        x = x_new
    
    # 达到最大迭代次数，返回当前估计
    return float(x)

def q_hampel_robust_algorithm(data, scheme="strict"):
    """
    基于我们自己的Q方法和修正Hampel方法的稳健统计方法
    """
    try:
        # 确保数据是numpy数组
        data_array = np.asarray(data, dtype=float)
        n = len(data_array)
        
        # 处理边界情况
        if n == 0:
            return _create_empty_result(scheme)
        elif n == 1:
            return _create_single_point_result(data_array[0], scheme)
        
        # 1. 使用Q方法计算稳健标准差
        robust_std = perform_Q_estimate_corrected(data_array)
        
        # 2. 使用修正的Hampel方法计算稳健平均值
        robust_mean = hampel_robust_mean(data_array)
        
        # 3. 计算正常值范围和识别离群值
        lower_limit = robust_mean - 3 * robust_std
        upper_limit = robust_mean + 3 * robust_std
        
        outliers_mask = (data_array < lower_limit) | (data_array > upper_limit)
        
        # 4. 分离正常数据和离群值
        clean_data = data_array[~outliers_mask].tolist()
        outliers = data_array[outliers_mask].tolist()
        
        # 5. 计算Z比分
        if robust_std > 1e-12:
            Z_scores_high_precision = ((data_array - robust_mean) / robust_std).tolist()
            Z_scores_rounded = np.round(Z_scores_high_precision, 2).tolist()
        else:
            Z_scores_high_precision = [0.0] * n
            Z_scores_rounded = [0.0] * n
        
        # 6. 根据计算方案格式化结果
        return _format_q_hampel_results(
            data_array, robust_mean, robust_std, clean_data, outliers, 
            Z_scores_high_precision, Z_scores_rounded,
            lower_limit, upper_limit, scheme
        )
        
    except Exception as e:
        # 如果计算失败，使用回退方法
        import traceback
        error_msg = f"Q/Hampel方法计算失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return _fallback_method(data, scheme, error_msg)

def _format_q_hampel_results(data, robust_mean, robust_std, clean_data, outliers, 
                           Z_scores_high_precision, Z_scores_rounded,
                           lower_limit, upper_limit, scheme):
    """格式化Q/Hampel法结果"""
    
    # 检测数据的小数位数
    decimal_places = detect_decimal_places(data)
    
    if scheme == "presentation":
        # 规范展示方案
        formatted_robust_mean = round(robust_mean, decimal_places)
        formatted_robust_std = round(robust_std, 3)
        
        # 使用格式化后的值重新计算Z比分（用于展示）
        if formatted_robust_std > 1e-12:
            Z_scores_high_precision = ((data - formatted_robust_mean) / formatted_robust_std).tolist()
            Z_scores_rounded = np.round(Z_scores_high_precision, 2).tolist()
        else:
            Z_scores_high_precision = [0.0] * len(data)
            Z_scores_rounded = [0.0] * len(data)
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_robust_mean})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差。"
        
        display_mean = formatted_robust_mean
        display_std = formatted_robust_std
        
        # 在规范展示方案中，初始中位数和MAD也按原始数据小数位数格式化
        initial_median = np.median(data)
        formatted_initial_median = round(initial_median, decimal_places)
        
        # 计算MAD（中位数绝对偏差）
        residuals = data - initial_median
        mad = np.median(np.abs(residuals))
        formatted_mad = round(mad, decimal_places)
        
    else:
        # 严格计算方案
        formatting_note = "使用严格计算方案：保留完整计算精度。"
        
        display_mean = robust_mean
        display_std = robust_std
        
        # 在严格计算方案中，显示6位小数
        initial_median = np.median(data)
        formatted_initial_median = initial_median
        
        residuals = data - initial_median
        mad = np.median(np.abs(residuals))
        formatted_mad = mad
    
    # 格式化Z比分为两位小数用于显示
    formatted_Z_scores = format_z_scores(Z_scores_rounded)
    
    # 为每个数据点生成分类
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
        'original_std': float(robust_std),
        'initial_median': float(formatted_initial_median),
        'mad': float(formatted_mad)
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
        'original_std': 0.0,
        'initial_median': 0.0,
        'mad': 0.0
    }

def _create_single_point_result(value, scheme):
    """创建单数据点结果"""
    decimal_places = detect_decimal_places([value])
    formatted_value = round(value, decimal_places) if scheme == "presentation" else value
    
    return {
        'robust_mean': float(formatted_value),
        'robust_std': 0.0,
        'clean_data': [float(value)],
        'outliers': [],
        'Z_scores_high_precision': [0.0],
        'Z_scores_rounded': [0.0],
        'formatted_Z_scores': ["0.00"],
        'z_score_classifications': ["满意"],
        'method_name': 'Q/Hampel法（单数据点）',
        'lower_limit': float(formatted_value),
        'upper_limit': float(formatted_value),
        'weights': [1.0],
        'iterations': 0,
        'formatting_note': "只有一个数据点，无法计算标准差",
        'calculation_scheme': scheme,
        'decimal_places': decimal_places,
        'original_mean': float(value),
        'original_std': 0.0,
        'initial_median': float(formatted_value),
        'mad': 0.0
    }

# 添加缺失的回退方法
def _fallback_method(data, scheme, error_message=""):
    """回退方法 - 使用传统统计量"""
    data_array = np.asarray(data, dtype=float)
    mean_val = float(np.mean(data_array))
    std_val = float(np.std(data_array, ddof=1)) if len(data_array) > 1 else 0.0
    
    # 计算Z比分
    if std_val > 0:
        Z_scores_high_precision = ((data_array - mean_val) / std_val).tolist()
        Z_scores_rounded = np.round(Z_scores_high_precision, 2).tolist()
    else:
        Z_scores_high_precision = [0.0] * len(data_array)
        Z_scores_rounded = [0.0] * len(data_array)
    
    # 格式化Z比分显示
    formatted_Z_scores = format_z_scores(Z_scores_rounded)
    
    # 为每个数据点生成分类
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
        'original_std': std_val,
        'initial_median': mean_val,
        'mad': 0.0
    }

# =============================================
# 统一结果显示组件 - 修改版本：采用2行4列布局
# =============================================

def display_core_results(results, method):
    """显示核心结果 - 修改版本：采用统一的2行4列布局"""
    st.subheader("📊 统计分析结果")
    
    # 根据计算方案格式化显示
    decimal_places = results.get('decimal_places', 0)
    calculation_scheme = results.get('calculation_scheme', 'strict')
    
    # 第一行：4个核心指标
    col1, col2, col3, col4 = st.columns(4)
    
    # 第一个指标：稳健平均值
    with col1:
        if calculation_scheme == "presentation":
            # 规范展示方案：使用原始数据小数位数
            formatted_mean = round(results['robust_mean'], decimal_places)
            if decimal_places == 0:
                display_value = f"{int(formatted_mean)}"
            else:
                display_value = f"{formatted_mean}"
            st.metric("稳健平均值", display_value)
        else:
            # 严格计算方案：显示6位小数
            st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
    
    # 第二个指标：稳健标准差
    with col2:
        if calculation_scheme == "presentation":
            # 规范展示方案：保留3位小数
            formatted_std = round(results['robust_std'], 3)
            st.metric("稳健标准差", f"{formatted_std}")
        else:
            # 严格计算方案：显示6位小数
            st.metric("稳健标准差", f"{results['robust_std']:.6f}")
    
    # 第三个指标：正常值下限
    with col3:
        if calculation_scheme == "presentation":
            # 规范展示方案：使用原始数据小数位数
            formatted_lower = round(results['lower_limit'], decimal_places)
            if decimal_places == 0:
                display_value = f"{int(formatted_lower)}"
            else:
                display_value = f"{formatted_lower}"
            st.metric("正常值下限", display_value)
        else:
            # 严格计算方案：显示6位小数
            st.metric("正常值下限", f"{results['lower_limit']:.6f}")
    
    # 第四个指标：正常值上限
    with col4:
        if calculation_scheme == "presentation":
            # 规范展示方案：使用原始数据小数位数
            formatted_upper = round(results['upper_limit'], decimal_places)
            if decimal_places == 0:
                display_value = f"{int(formatted_upper)}"
            else:
                display_value = f"{formatted_upper}"
            st.metric("正常值上限", display_value)
        else:
            # 严格计算方案：显示6位小数
            st.metric("正常值上限", f"{results['upper_limit']:.6f}")
    
    # 第二行：方法特定指标
    st.markdown('<div class="aligned-metrics">', unsafe_allow_html=True)
    
    # 根据方法显示不同的第二行指标
    if method == "迭代稳健统计法":
        # 迭代法：迭代次数、收敛状态、离群值数量、尺度因子
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            iterations = results.get('iterations', 0)
            st.metric("迭代次数", f"{iterations}")
        
        with col2:
            converged = results.get('converged', False)
            converged_text = "是" if converged else "否"
            st.metric("收敛状态", converged_text)
        
        with col3:
            outliers_count = len(results.get('outliers', []))
            st.metric("离群值数量", f"{outliers_count}")
        
        with col4:
            # 从全局变量获取尺度因子
            if 'k_value' in globals():
                st.metric("尺度因子(k)", f"{k_value}")
            else:
                st.metric("尺度因子(k)", "1.5")
    
    elif method == "四分位稳健统计法":
        # 四分位法：Q1、Q3、IQR、NIQR
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if calculation_scheme == "presentation":
                # 规范展示方案：按原始数据小数位数格式化
                formatted_q1 = round(results['q1'], decimal_places)
                if decimal_places == 0:
                    display_value = f"{int(formatted_q1)}"
                else:
                    display_value = f"{formatted_q1}"
                st.metric("下四分位数(Q1)", display_value)
            else:
                # 严格计算方案：显示6位小数
                st.metric("下四分位数(Q1)", f"{results['q1']:.6f}")
        
        with col2:
            if calculation_scheme == "presentation":
                # 规范展示方案：按原始数据小数位数格式化
                formatted_q3 = round(results['q3'], decimal_places)
                if decimal_places == 0:
                    display_value = f"{int(formatted_q3)}"
                else:
                    display_value = f"{formatted_q3}"
                st.metric("上四分位数(Q3)", display_value)
            else:
                # 严格计算方案：显示6位小数
                st.metric("上四分位数(Q3)", f"{results['q3']:.6f}")
        
        with col3:
            if calculation_scheme == "presentation":
                # 规范展示方案：按原始数据小数位数格式化
                formatted_iqr = round(results['iqr'], decimal_places)
                if decimal_places == 0:
                    display_value = f"{int(formatted_iqr)}"
                else:
                    display_value = f"{formatted_iqr}"
                st.metric("四分位距(IQR)", display_value)
            else:
                # 严格计算方案：显示6位小数
                st.metric("四分位距(IQR)", f"{results['iqr']:.6f}")
        
        with col4:
            if calculation_scheme == "presentation":
                # 规范展示方案：NIQR保留3位小数
                formatted_niqr = round(results['niqr'], 3)
                st.metric("标准化四分位距(NIQR)", f"{formatted_niqr}")
            else:
                # 严格计算方案：显示6位小数
                st.metric("标准化四分位距(NIQR)", f"{results['niqr']:.6f}")
    
    elif method == "Q/Hampel法":
        # Q/Hampel法：初始中位数、MAD值、离群值数量、权重平均值
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if calculation_scheme == "presentation":
                # 规范展示方案：按原始数据小数位数格式化
                initial_median = results.get('initial_median', results['robust_mean'])
                formatted_initial_median = round(initial_median, decimal_places)
                if decimal_places == 0:
                    display_value = f"{int(formatted_initial_median)}"
                else:
                    display_value = f"{formatted_initial_median}"
                st.metric("初始中位数", display_value)
            else:
                # 严格计算方案：显示6位小数
                initial_median = results.get('initial_median', results['robust_mean'])
                st.metric("初始中位数", f"{initial_median:.6f}")
        
        with col2:
            if calculation_scheme == "presentation":
                # 规范展示方案：按原始数据小数位数格式化
                mad = results.get('mad', 0)
                formatted_mad = round(mad, decimal_places)
                if decimal_places == 0:
                    display_value = f"{int(formatted_mad)}"
                else:
                    display_value = f"{formatted_mad}"
                st.metric("MAD值", display_value)
            else:
                # 严格计算方案：显示6位小数
                mad = results.get('mad', 0)
                st.metric("MAD值", f"{mad:.6f}")
        
        with col3:
            outliers_count = len(results.get('outliers', []))
            st.metric("离群值数量", f"{outliers_count}")
        
        with col4:
            # 计算权重平均值（如果可用）
            if 'weights' in results and results['weights']:
                weights = results['weights']
                weighted_avg = np.average(results['clean_data'], weights=weights[:len(results['clean_data'])]) if len(results['clean_data']) > 0 else 0
                if calculation_scheme == "presentation":
                    formatted_weighted_avg = round(weighted_avg, decimal_places)
                    if decimal_places == 0:
                        display_value = f"{int(formatted_weighted_avg)}"
                    else:
                        display_value = f"{formatted_weighted_avg}"
                    st.metric("权重平均值", display_value)
                else:
                    st.metric("权重平均值", f"{weighted_avg:.6f}")
            else:
                st.metric("权重平均值", "N/A")
    
    elif method == "Z比分计算模块":
        # Z比分计算模块：输入的平均值、输入的标准差、离群值数量、Z比分范围
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 显示输入的稳健平均值
            robust_mean_val = results['robust_mean']
            if calculation_scheme == "presentation":
                formatted_mean = round(robust_mean_val, decimal_places)
                if decimal_places == 0:
                    display_value = f"{int(formatted_mean)}"
                else:
                    display_value = f"{formatted_mean}"
                st.metric("输入稳健平均值", display_value)
            else:
                st.metric("输入稳健平均值", f"{robust_mean_val:.6f}")
        
        with col2:
            # 显示输入的稳健标准差
            robust_std_val = results['robust_std']
            if calculation_scheme == "presentation":
                formatted_std = round(robust_std_val, 3)
                st.metric("输入稳健标准差", f"{formatted_std}")
            else:
                st.metric("输入稳健标准差", f"{robust_std_val:.6f}")
        
        with col3:
            outliers_count = len(results.get('outliers', []))
            st.metric("离群值数量", f"{outliers_count}")
        
        with col4:
            # 计算Z比分范围
            z_scores = results.get('Z_scores_rounded', [])
            if z_scores:
                min_z = min(z_scores)
                max_z = max(z_scores)
                st.metric("Z比分范围", f"[{min_z:.2f}, {max_z:.2f}]")
            else:
                st.metric("Z比分范围", "N/A")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_z_score_analysis(results):
    """统一显示Z比分分析"""
    # Z比分分类统计
    st.subheader("📊 Z比分分析")
    
    z_scores_data = results['Z_scores_rounded']
    if z_scores_data is not None:
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

def display_detailed_results(results, method, data, original_labels=None):
    """显示详细结果 - 可折叠"""
    with st.expander("📋 详细结果", expanded=False):
        # 格式化说明 - 仅在详细结果中保留
        if 'formatting_note' in results:
            st.info(f"💡 {results['formatting_note']}")
        
        # 离群值显示
        if len(results['outliers']) > 0:
            outliers_list = results['outliers']
            if hasattr(outliers_list, '__iter__') and not isinstance(outliers_list, str):
                try:
                    outliers_list = [float(x) for x in outliers_list]
                    outliers_list = sorted(outliers_list)
                    
                    st.write(f"**离群值** ({len(outliers_list)}个):")
                    
                    # 按Z比分分类显示离群值
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

# =============================================
# 统计量表显示函数 - 修改版本：只展示到"极差"
# =============================================

def display_statistics_table(results, method, data, input_method, original_labels=None):
    """显示统计量表 - 修改版本：只展示到'极差'"""
    st.subheader("📊 统计量表")
    
    # 根据输入方式选择正确的数据源
    if input_method == "文件上传":
        current_original_data = st.session_state.file_original_data
        current_decimal_info = st.session_state.file_decimal_info
        current_blank_count = st.session_state.file_blank_count
    elif input_method == "带编号数据输入":
        # 对于两列数据，需要特殊处理
        current_original_data = [value for _, value in st.session_state.valid_pairs]
        current_decimal_info = st.session_state.two_column_decimal_info
        current_blank_count = 0  # 两列数据已经过滤了无效数据
    else:
        # 手动输入或示例数据
        current_original_data = st.session_state.original_data
        current_decimal_info = st.session_state.decimal_info
        current_blank_count = st.session_state.blank_count
    
    # 获取检测到的小数位数 - 使用正确的数据源
    detected_decimal_places = results.get('decimal_places', 2)
    if current_decimal_info and 'detected_decimal_places' in current_decimal_info:
        detected_decimal_places = current_decimal_info['detected_decimal_places']
    
    # 确保detected_decimal_places是一个整数
    if detected_decimal_places is None:
        detected_decimal_places = 2
    
    # 辅助函数：根据小数位数格式化数字
    def format_number(value, decimal_places):
        """根据小数位数格式化数字"""
        if value is None or pd.isna(value):
            return None
        if decimal_places == 0:
            return int(value)  # 如果是整数，返回整数形式
        return round(value, decimal_places)
    
    # 计算统计量 - 只计算到极差，删除正常值下限和上限
    total_data_count = len(current_original_data) if current_original_data else len(data)
    actual_analyzable_count = len(data)
    blank_data_count = current_blank_count if current_blank_count else 0
    
    # 创建统计量表 - 只展示到极差
    stats_data = {
        '统计量名称': ['总数据数', '实际可分析数据数', '空白数据数', '稳健平均值', '稳健标准差', 
                     '最小值', '最大值', '极差'],
        '数值': [
            total_data_count,
            actual_analyzable_count,
            blank_data_count,
            format_number(results['robust_mean'], detected_decimal_places),
            format_number(results['robust_std'], 3),  # 标准差保持3位
            format_number(np.min(data), detected_decimal_places) if len(data) > 0 else 0,
            format_number(np.max(data), detected_decimal_places) if len(data) > 0 else 0,
            format_number(np.max(data) - np.min(data), detected_decimal_places) if len(data) > 0 else 0
        ]
    }
    
    # 根据方法添加特定的统计量
    if method == "四分位稳健统计法":
        stats_data['统计量名称'].extend(['下四分位数(Q1)', '上四分位数(Q3)', '四分位距(IQR)', '标准化四分位距(NIQR)'])
        stats_data['数值'].extend([
            format_number(results['q1'], detected_decimal_places),
            format_number(results['q3'], detected_decimal_places),
            format_number(results['iqr'], detected_decimal_places),
            format_number(results['niqr'], 3)  # NIQR保留3位小数
        ])
    elif method == "Q/Hampel法":
        stats_data['统计量名称'].extend(['初始中位数', 'MAD值'])
        stats_data['数值'].extend([
            format_number(results.get('initial_median', results['robust_mean']), detected_decimal_places),
            format_number(results.get('mad', 0), detected_decimal_places)
        ])
    
    # 创建DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # 显示表格
    st.dataframe(stats_df, use_container_width=True)
    
    return stats_df

# =============================================
# 主程序分析部分
# =============================================

# 执行分析
if data is not None and len(data) > 0:
    try:
        # 确保data是有效的数值数组
        if isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            st.error("❌ 数据格式无效")
            st.stop()
        
        # 检查数据是否包含有效数值
        if len(data) == 0:
            st.error("❌ 没有有效数据可供分析")
            st.stop()
        
        st.markdown("---")
        st.subheader(f"📈 {method}分析结果")
        
        # 从会话状态获取计算方案
        calculation_scheme = st.session_state.get('calculation_scheme', '严格计算方案')
        
        # 显示当前选择的计算方案
        scheme_display = "规范展示方案" if calculation_scheme == "规范展示方案" else "严格计算方案"
        st.info(f"当前使用: **{scheme_display}**")
        
        # 特别处理：Z比分计算模块的输入验证
        if method == "Z比分计算模块":
            # 检查稳健统计量输入
            try:
                # 直接从主界面的输入框获取值
                robust_mean_val = float(st.session_state.robust_mean_input)
                robust_std_val = float(st.session_state.robust_std_input)
                
                # 验证稳健标准差必须大于0
                if robust_std_val <= 0:
                    st.error("❌ 稳健标准差必须大于0")
                    st.stop()
                
                # 显示使用的参数
                st.success(f"✅ 使用参数: 稳健平均值 = {robust_mean_val}, 稳健标准差 = {robust_std_val}")
                
            except ValueError:
                st.error("❌ 稳健统计量格式错误，请输入有效的数字")
                st.stop()
            except KeyError:
                st.error("❌ 请先输入稳健统计量")
                st.stop()
        
        # 执行稳健统计分析
        with st.spinner(f"正在执行{method}分析..."):
            # 将方案选择转换为参数
            scheme_param = "presentation" if calculation_scheme == "规范展示方案" else "strict"
            
            # 根据选择的方法执行分析
            if method == "迭代稳健统计法":
                results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme=scheme_param)
            elif method == "四分位稳健统计法":
                results = quartile_robust_algorithm(data, scheme=scheme_param)
            elif method == "Q/Hampel法":
                results = q_hampel_robust_algorithm(data, scheme=scheme_param)
            elif method == "Z比分计算模块":
                # 使用Z比分计算方法
                results = z_score_calculation_algorithm(data, robust_mean_val, robust_std_val, scheme=scheme_param)

        # 显示核心结果（2行4列布局）
        display_core_results(results, method)
        
        # 显示Z比分分析
        display_z_score_analysis(results)
        
        # 显示详细结果（可折叠）
        display_detailed_results(results, method, data)
        
        # 显示统计量表（只展示到极差）
        stats_df = display_statistics_table(results, method, data, input_method)
        
        # =============================================
        # 数据可视化
        # =============================================
        st.subheader("📊 数据可视化")
        
        # 获取原始标签
        original_labels = None
        if input_method == "带编号数据输入" and st.session_state.label_data_pairs:
            original_labels = [pair[0] for pair in st.session_state.valid_pairs]
        else:
            # 生成三位数字标签
            n_points = len(data)
            original_labels = [f"{i+1:03d}" for i in range(n_points)]
        
        # 显示Z比分图表
        fig = create_z_score_chart(results, original_labels)
        if fig is not None:
            st.pyplot(fig)
        
        # =============================================
        # 方案比较功能（可选）
        # =============================================
        if show_scheme_comparison and method != "Z比分计算模块":
            st.markdown("---")
            st.subheader("🔍 计算方案对比")
            
            # 根据当前选择的方法执行两种方案的计算
            with st.spinner("正在计算方案对比..."):
                if method == "迭代稳健统计法":
                    strict_results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme="strict")
                    presentation_results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme="presentation")
                elif method == "四分位稳健统计法":
                    strict_results = quartile_robust_algorithm(data, scheme="strict")
                    presentation_results = quartile_robust_algorithm(data, scheme="presentation")
                else:  # Q/Hampel法
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
            
            # 显示方案差异说明
            st.info("""
            **方案差异说明:**
            - **严格计算方案**: 使用完整计算精度，确保计算准确性
            - **规范展示方案**: 稳健平均值与原始数据小数位数一致，结果更规范但可能引入微小误差
            """)
    
        # =============================================
        # 导出结果模块
        # =============================================
        st.subheader("💾 导出结果")
        
        # 辅助函数：根据小数位数格式化数字
        def format_number(value, decimal_places):
            """根据小数位数格式化数字"""
            if value is None or pd.isna(value):
                return None
            if decimal_places == 0:
                return int(value)  # 如果是整数，返回整数形式
            return round(value, decimal_places)
        
        # 根据输入方式选择正确的数据源
        if input_method == "文件上传":
            current_original_data = st.session_state.file_original_data
            current_decimal_info = st.session_state.file_decimal_info
            current_blank_count = st.session_state.file_blank_count
        elif input_method == "带编号数据输入":
            # 对于两列数据，需要特殊处理
            current_original_data = [value for _, value in st.session_state.valid_pairs]
            current_decimal_info = st.session_state.two_column_decimal_info
            current_blank_count = 0  # 两列数据已经过滤了无效数据
        else:
            # 手动输入或示例数据
            current_original_data = st.session_state.original_data
            current_decimal_info = st.session_state.decimal_info
            current_blank_count = st.session_state.blank_count
        
        # 获取检测到的小数位数 - 使用正确的数据源
        detected_decimal_places = results.get('decimal_places', 2)
        if current_decimal_info and 'detected_decimal_places' in current_decimal_info:
            detected_decimal_places = current_decimal_info['detected_decimal_places']
        
        # 确保detected_decimal_places是一个整数
        if detected_decimal_places is None:
            detected_decimal_places = 2
        
        # 创建结果DataFrame - 支持原始标签
        result_data = []
        
        if input_method == "带编号数据输入" and st.session_state.label_data_pairs:
            # 两列数据输入：使用用户提供的原始标签
            valid_data_count = 0
            for label, value in st.session_state.label_data_pairs:
                # 检查是否有对应的Z分数（仅对有效数据）
                z_score = None
                if value is not None and valid_data_count < len(results['formatted_Z_scores']):
                    z_score = results['formatted_Z_scores'][valid_data_count]
                    classification = results['z_score_classifications'][valid_data_count]
                    valid_data_count += 1
                
                # 使用检测到的小数位数格式化
                formatted_value = format_number(value, detected_decimal_places)
                # 使用新的格式化函数确保显示两位小数
                formatted_z_score = format_z_score_display(z_score)
                
                result_data.append({
                    '标签原始标号': label,  # 使用用户提供的标签
                    '输入数据': formatted_value,
                    'Z比分数': formatted_z_score,
                    '分类结果': classification
                })
            
            total_data_count = len(st.session_state.label_data_pairs)
            blank_data_count = sum(1 for _, value in st.session_state.label_data_pairs if value is None)
            actual_analyzable_count = total_data_count - blank_data_count
            
        else:
            # 其他输入方式：使用自动生成的三位数字标签
            valid_data_count = 0
            
            # 使用正确的原始数据源
            if current_original_data:
                for i, value in enumerate(current_original_data):
                    original_label = f"{str(i+1).zfill(3)}"  # 001, 002, ...
                    
                    if value is not None:  # 有效数据
                        z_score = results['formatted_Z_scores'][valid_data_count] if valid_data_count < len(results['formatted_Z_scores']) else None
                        classification = results['z_score_classifications'][valid_data_count] if valid_data_count < len(results['z_score_classifications']) else "未知"
                        # 使用检测到的小数位数格式化
                        formatted_value = format_number(value, detected_decimal_places)
                        # 使用新的格式化函数确保显示两位小数
                        formatted_z_score = format_z_score_display(z_score)
                        
                        result_data.append({
                            '标签原始标号': original_label,
                            '输入数据': formatted_value,
                            'Z比分数': formatted_z_score,
                            '分类结果': classification
                        })
                        valid_data_count += 1
                    else:  # 空白数据
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
                # 如果没有原始数据信息，使用简单处理
                for i, value in enumerate(data):
                    original_label = f"{str(i+1).zfill(3)}"
                    z_score = results['formatted_Z_scores'][i] if i < len(results['formatted_Z_scores']) else None
                    classification = results['z_score_classifications'][i] if i < len(results['z_score_classifications']) else "未知"
                    
                    # 使用检测到的小数位数格式化
                    formatted_value = format_number(value, detected_decimal_places)
                    # 使用新的格式化函数确保显示两位小数
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
        
        # 确保结果数据数量与Z分数数量一致
        if len(result_data) != len(results['formatted_Z_scores']):
            st.warning(f"⚠️ 数据数量不匹配: 结果数据({len(result_data)}) vs Z分数({len(results['formatted_Z_scores'])})")
            # 调整结果数据以匹配Z分数数量
            if len(result_data) > len(results['formatted_Z_scores']):
                result_data = result_data[:len(results['formatted_Z_scores'])]
            else:
                # 如果结果数据较少，补充空数据
                while len(result_data) < len(results['formatted_Z_scores']):
                    result_data.append({
                        '标签原始标号': f"{str(len(result_data)+1).zfill(3)}",
                        '输入数据': None,
                        'Z比分数': "",
                        '分类结果': ""
                    })
        
        result_df = pd.DataFrame(result_data)
        
        # 在文本报告开头添加方案说明和小数位数说明
        scheme_text = "严格计算方案" if calculation_scheme == "严格计算方案" else "规范展示方案"
        report = f"""                
{method}分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
工具版本: 稳健统计分析工具 (Robust Statistical Analysis Tool)
计算方案: {scheme_text}
数据小数位数: {detected_decimal_places}位（基于输入数据的最大小数位数）

数据概览:
--------
总数据点数: {total_data_count}
实际可分析数据数: {actual_analyzable_count}
空白数据数: {blank_data_count}

数据表格:
--------
标签原始标号\t输入数据\tZ比分数\t分类结果
"""
        
        # 添加数据行 - 确保数值格式与预览一致
        for i in range(len(result_df)):
            row = result_df.iloc[i]
            # 处理输入数据格式 - 使用与预览相同的格式化
            if pd.isna(row['输入数据']):
                input_data = ""
            else:
                # 使用检测到的小数位数格式化
                if detected_decimal_places == 0:
                    input_data = f"{int(row['输入数据'])}"  # 整数格式
                else:
                    input_data = f"{row['输入数据']:.{detected_decimal_places}f}"
            
            # 处理Z比分数格式 - 使用与预览相同的格式化
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
        
        # Z比分数分类统计
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
            # Excel导出
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # 数据表格工作表
                result_df.to_excel(writer, sheet_name='分析数据', index=False)
                
                # 统计量工作表
                stats_df.to_excel(writer, sheet_name='统计摘要', index=False)
                
                # 详细信息工作表
                detail_data = {
                    '项目': ['分析方法', '总数据点数', '实际可分析数据数', '空白数据数', 
                           '稳健平均值', '稳健标准差', '离群值数量', '正常值下限', '正常值上限',
                           '数据小数位数'],
                    '数值': [method, total_data_count, actual_analyzable_count, blank_data_count,
                           format_number(results['robust_mean'], detected_decimal_places),
                           format_number(results['robust_std'], 3),  # 标准差保持3位
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
            # JSON导出
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
            # CSV导出
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV",
                data=csv_data,
                file_name=f"{method}_分析结果.csv",
                mime="text/csv",
                help="下载CSV格式的分析结果表格"
            )
        
        with export_col4:
            # 文本报告导出
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
        
    except Exception as e:
        st.error(f"❌ 统计分析过程中发生错误: {str(e)}")
        st.info("💡 这可能是因为数据特征不适合所选的分析方法，请尝试其他统计方法或检查数据质量")

else:
    st.info("👆 请先输入或上传数据以开始分析")

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

# =============================================
# 添加页面底部空白，确保用户反馈模块完整显示
# =============================================
st.markdown('<div class="page-bottom-padding"></div>', unsafe_allow_html=True)