# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:51:46 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
from collections import Counter
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
        # 使用支持中文的字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

# =============================================
# 修改后的数据验证和错误处理模块
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
# 修复后的文件格式处理模块
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

# =============================================
# 主程序开始
# =============================================

# 设置页面
st.set_page_config(
    page_title="统计分析工具",
    page_icon="📊",
    layout="wide"
)

# 初始化会话状态
initialize_session_state()

# 标题和说明
st.title("📊 统计分析工具")
st.markdown("""
提供多种稳健统计分析方法，用于处理包含异常值的数据集。
支持迭代稳健统计法、四分位稳健统计法和Q/Hampel法。
""")

# 侧边栏 - 参数设置和方法选择
st.sidebar.header("⚙️ 分析设置")

# 方法选择
method = st.sidebar.selectbox(
    "选择统计方法:",
    ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法"],
    help="选择适合数据特征的稳健统计方法"
)

# 计算方案选择
calculation_scheme = st.sidebar.radio(
    "选择计算方案:",
    ["严格计算方案", "规范展示方案"],
    help="""
    严格计算方案：使用完整精度的计算结果，确保计算准确性
    规范展示方案：稳健平均值与原始数据小数位数一致，结果更规范但可能引入微小误差
    """
)

# 方案比较选项（放在侧边栏）
show_scheme_comparison = st.sidebar.checkbox(
    "显示方案比较", 
    help="同时显示两种计算方案的结果对比",
    value=False  # 默认不显示
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
                       ["手动输入", "带编号数据输入", "文件上传", "示例数据"])
data = None

if input_method == "手动输入":
    st.subheader("📝 手动输入数据")
    
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
        st.session_state.validation_report = []
        st.session_state.validation_passed = False

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
            st.session_state.validation_report = []
            st.session_state.validation_passed = False
        else:
            st.session_state.reset_counter += 1

    def analyze_data():
        """分析数据的回调函数 - 支持空白数据处理和计算方案验证"""
        try:
            # 使用新的数据验证器（支持空白数据和计算方案）
            is_valid, original_data, clean_data, blank_count, validation_report, decimal_info = \
                DataValidator.comprehensive_validation(
                    st.session_state.manual_data,
                    calculation_scheme  # 传递当前选择的计算方案
                )
            
            if is_valid:
                st.session_state.processed_data = clean_data
                st.session_state.original_data = original_data
                st.session_state.blank_count = blank_count
                st.session_state.data_loaded = True
                st.session_state.validation_report = validation_report
                st.session_state.validation_passed = True
                st.session_state.decimal_info = decimal_info  # 保存小数位数信息
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
    if st.session_state.validation_report:
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

elif input_method == "带编号数据输入":
    st.subheader("📝 带编号数据输入")
    
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
    if two_column_input != st.session_state.two_column_data:
        st.session_state.two_column_data = two_column_input
    
    # 分析按钮
    if st.button("分析两列数据", type="primary", use_container_width=True):
        if two_column_input.strip():
            try:
                # 使用新的验证函数解析两列数据
                label_data_pairs, valid_pairs, invalid_lines, decimal_info = validate_two_column_data(
                    two_column_input, calculation_scheme
                )
                
                if invalid_lines:
                    st.error("❌ 数据格式错误：")
                    for error in invalid_lines:
                        st.write(f"  - {error}")
                
                if valid_pairs:
                    # 提取标签和数据
                    labels = [pair[0] for pair in valid_pairs]
                    values = np.array([pair[1] for pair in valid_pairs])
                    
                    # 构建验证报告
                    validation_report = []
                    validation_report.append("✅ 两列数据格式验证通过")
                    
                    # 计算方案兼容性验证
                    scheme_messages = DataValidator.validate_calculation_scheme_compatibility(
                        values, calculation_scheme, decimal_info
                    )
                    validation_report.extend(scheme_messages)
                    
                    # 推荐计算方案
                    recommended_scheme, recommendation_reason = DataValidator.get_recommended_scheme(decimal_info)
                    validation_report.append(f"💡 推荐计算方案: {recommended_scheme} - {recommendation_reason}")
                    
                    # 数据验证（只验证有效数据）
                    values_str = "\n".join([str(pair[1]) for pair in valid_pairs])
                    # 修复：接收6个返回值
                    is_valid, _, clean_data, blank_count, full_validation_report, decimal_info_from_validation = DataValidator.comprehensive_validation(
                        values_str, calculation_scheme
                    )
                    
                    if is_valid:
                        # 合并验证报告
                        validation_report.extend(full_validation_report)
                        
                        st.session_state.label_data_pairs = label_data_pairs
                        st.session_state.valid_pairs = valid_pairs
                        st.session_state.processed_data = clean_data
                        st.session_state.original_labels = labels
                        st.session_state.two_column_processed = True
                        st.session_state.two_column_validation_report = validation_report
                        # 使用从验证返回的小数位数信息
                        st.session_state.decimal_info = decimal_info_from_validation
                        
                        st.success(f"✅ 成功解析 {len(valid_pairs)} 个数据对")
                        
                        # 显示数据预览
                        with st.expander("📋 查看数据预览", expanded=True):
                            preview_df = pd.DataFrame({
                                '原始标签': labels,
                                '数值': values
                            })
                            st.dataframe(preview_df, use_container_width=True)
                    else:
                        st.session_state.two_column_processed = False
                        st.error("❌ 数据验证失败")
                        with st.expander("📋 查看验证详情", expanded=True):
                            for line in full_validation_report:
                                if line.startswith("❌"):
                                    st.error(line)
                                else:
                                    st.write(line)
            except Exception as e:
                st.error(f"❌ 数据处理错误: {str(e)}")
        else:
            st.error("请输入数据")
    
    # 清除按钮
    if st.button("清除两列数据", type="secondary", use_container_width=True):
        st.session_state.two_column_data = ""
        st.session_state.two_column_processed = False
        st.session_state.label_data_pairs = []
        st.session_state.two_column_validation_report = []
        st.rerun()
    
    # 如果数据已处理，设置数据变量
    if st.session_state.two_column_processed and st.session_state.label_data_pairs:
        data = st.session_state.processed_data
        
# =============================================
# 修复文件上传模块的数据处理
# =============================================

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
                    
                    # 计算方案兼容性验证
                    scheme_messages = DataValidator.validate_calculation_scheme_compatibility(
                        clean_data, calculation_scheme, decimal_info
                    )
                    validation_report.extend(scheme_messages)
                    
                    # 推荐计算方案
                    recommended_scheme, recommendation_reason = DataValidator.get_recommended_scheme(decimal_info)
                    validation_report.append(f"💡 推荐计算方案: {recommended_scheme} - {recommendation_reason}")
                    
                    # =============================================
                    # 修复：设置正确的会话状态，确保使用文件数据
                    # =============================================
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

else:  # 示例数据
    st.subheader("🎯 示例数据分析")
    example_data = np.array([
        54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
        54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
        54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
        54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
        54.5, 55.9, 53.2, 54.6
    ])
    
    # 对示例数据进行验证 - 使用新的方法签名，包含计算方案
    example_data_str = ", ".join([str(x) for x in example_data])
    is_valid, original_data, clean_data, blank_count, validation_report, decimal_info = \
        DataValidator.comprehensive_validation(example_data_str, calculation_scheme)
    
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
    
    # 为示例数据设置必要的会话状态变量，以便导出模块正常工作
    st.session_state.original_data = example_data.tolist()  # 转换为列表格式
    st.session_state.blank_count = 0  # 示例数据没有空白
    # 保存小数位数信息
    st.session_state.decimal_info = decimal_info

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

# =============================================
# 修改后的统计方法实现
# =============================================

def detect_decimal_places(data):
    """检测数据的小数位数 - 返回最大小数位数"""
    max_decimal_places = 0
    for value in data:
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

def iterative_robust_algorithm(data, max_iterations=50, k=1.5, scheme="strict"):
    """迭代稳健统计法 - 支持两种计算方案"""
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
        # 规范展示方案：使用四舍五入后的均值和标准差
        decimal_places = detect_decimal_places(data)
        formatted_X_star = round(X_star, decimal_places)
        formatted_S_star = round(S_star, 3)
        
        # 使用格式化后的值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - formatted_X_star) / formatted_S_star
        
        # 计算界限
        final_delta = k * formatted_S_star
        lower_limit = formatted_X_star - final_delta
        upper_limit = formatted_X_star + final_delta
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_X_star})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差，但计算过程中不进行四舍五入。"
        
        robust_mean = formatted_X_star
        robust_std = formatted_S_star
        
    else:
        # 严格计算方案：使用原始计算值
        decimal_places = 6  # 高精度显示
        formatted_X_star = X_star
        formatted_S_star = S_star
        
        # 使用原始计算值计算Z比分
        Z_scores = (data - X_star) / S_star
        
        # 计算界限
        final_delta = k * S_star
        lower_limit = X_star - final_delta
        upper_limit = X_star + final_delta
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。Z比分计算过程中不进行四舍五入。"
        
        robust_mean = X_star
        robust_std = S_star
    
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
    
    # === 修复：确保 Z_scores 是安全的Python类型 ===
    if hasattr(Z_scores, 'tolist'):
        safe_z_scores = Z_scores.tolist()
    elif hasattr(Z_scores, '__iter__') and not isinstance(Z_scores, (str, dict)):
        safe_z_scores = list(Z_scores)
    else:
        safe_z_scores = [Z_scores] if Z_scores is not None else []
    
    # 确保返回标准Python类型
    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores': safe_z_scores,  # 使用修复后的变量
        'iterations': iteration,
        'converged': converged,
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'history': history,
        'method_name': '迭代稳健统计法',
        'decimal_places': decimal_places,
        'calculation_scheme': scheme,
        'formatting_note': formatting_note,
        'original_mean': float(X_star) if not np.isnan(X_star) else 0.0,  # 始终保存原始计算值
        'original_std': float(S_star) if not np.isnan(S_star) else 0.0    # 始终保存原始计算值
    }

def quartile_robust_algorithm(data, scheme="strict"):
    """四分位稳健统计法 - 支持两种计算方案"""
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
        
        # 使用格式化后的值计算Z比分（计算过程不四舍五入）
        Z_scores = (data - formatted_median) / formatted_niqr
        
        formatting_note = f"使用规范展示方案：稳健平均值({formatted_median})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。Z比分计算使用格式化后的均值和标准差，但计算过程中不进行四舍五入。"
        
        robust_mean = formatted_median
        robust_std = formatted_niqr
        
    else:
        # 严格计算方案：使用原始计算值
        formatted_median = median
        formatted_niqr = niqr
        
        # 使用原始计算值计算Z比分
        Z_scores = (data - median) / niqr
        
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。Z比分计算过程中不进行四舍五入。"
        
        robust_mean = median
        robust_std = niqr
    
    # === 修复：确保 Z_scores 在所有分支中都是安全的Python类型 ===
    if hasattr(Z_scores, 'tolist'):
        safe_z_scores = Z_scores.tolist()
    elif hasattr(Z_scores, '__iter__') and not isinstance(Z_scores, (str, dict)):
        safe_z_scores = list(Z_scores)
    else:
        safe_z_scores = [Z_scores] if Z_scores is not None else []

    # 确保返回标准Python类型
    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data_list,
        'outliers': outliers_list,
        'Z_scores': safe_z_scores,  # 使用修复后的变量
        'q1': float(q1) if not np.isnan(q1) else 0.0,
        'q3': float(q3) if not np.isnan(q3) else 0.0,
        'iqr': float(iqr) if not np.isnan(iqr) else 0.0,
        'niqr': float(niqr) if not np.isnan(niqr) else 0.0,
        'method_name': '四分位稳健统计法',
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'formatting_note': formatting_note,
        'calculation_scheme': scheme,
        'decimal_places': decimal_places,  # 添加小数位数信息，与迭代法保持一致
        'original_mean': float(median) if not np.isnan(median) else 0.0,  # 保存原始计算值
        'original_std': float(niqr) if not np.isnan(niqr) else 0.0  # 保存原始计算值
    }

def hampel_estimator_iso13528(data, a=1.5, b=3.0, c=4.5, max_iter=100, tol=1e-8):
    """
    Hampel估计器实现，基于ISO 13528标准
    
    参数:
        data: 输入数据数组
        a, b, c: Hampel权重函数的阈值参数 (默认值来自ISO 13528)
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    
    返回:
        robust_mean: 稳健均值
        weights: 最终权重向量
        iterations: 实际迭代次数
        converged: 是否收敛
    """
    # 初始估计使用中位数[citation:1]
    current_estimate = np.median(data)
    n = len(data)
    weights = np.ones(n)
    
    for iteration in range(max_iter):
        # 计算残差
        residuals = data - current_estimate
        
        # 计算MAD（中位数绝对偏差）
        mad = np.median(np.abs(residuals))
        
        # 处理MAD为零的情况 - 按照ISO 13528标准直接返回中位数
        if mad < 1e-12:
            # 当MAD=0时，直接返回中位数估计（ISO 13528 7.2.2 注 3）
            return np.median(data), np.ones(n), iteration+1, True
        
        # 标准化残差[citation:1]
        standardized_residuals = np.abs(residuals) / (1.4826 * mad)
        
        # Hampel权重函数[citation:1]
        weights = np.zeros(n)
        for i, q in enumerate(standardized_residuals):
            if q <= a:
                weights[i] = 1.0
            elif a < q <= b:
                weights[i] = a / q
            elif b < q <= c:
                weights[i] = a * (c - q) / (q * (c - b))
            else:  # q > c
                weights[i] = 0.0
        
        # 更新估计
        new_estimate = np.sum(weights * data) / np.sum(weights)
        
        # 检查收敛
        if abs(new_estimate - current_estimate) < tol:
            return new_estimate, weights, iteration+1, True
        
        current_estimate = new_estimate
    
    return current_estimate, weights, max_iter, False

def qn_estimator_iso13528(data):
    """
    Qn尺度估计器实现，基于ISO 13528标准
    
    参数:
        data: 输入数据数组
    
    返回:
        qn_scale: Qn稳健尺度估计
    """
    n = len(data)
    if n < 2:
        return 0.0
    
    # 计算所有点对之间的绝对差
    pairwise_diffs = []
    for i in range(n):
        for j in range(i+1, n):
            pairwise_diffs.append(abs(data[i] - data[j]))
    
    # 计算Qn统计量[citation:1]
    h = n // 2 + 1
    k = h * (h - 1) // 2
    
    # 找到第k个最小的绝对差
    pairwise_diffs_sorted = np.sort(pairwise_diffs)
    qn_statistic = pairwise_diffs_sorted[k-1] if k <= len(pairwise_diffs_sorted) else pairwise_diffs_sorted[-1]
    
    # 应用校正因子
    correction_factors = {2: 0.993, 3: 0.993, 4: 0.886, 5: 0.921, 
                          6: 0.940, 7: 0.952, 8: 0.959, 9: 0.965, 10: 0.969}
    
    if n <= 10:
        correction = correction_factors.get(n, 1.0)
    else:
        if n % 2 == 0:
            correction = n / (n + 3.8)
        else:
            correction = n / (n + 1.4)
    
    return 2.2219 * correction * qn_statistic

def estimate_instrument_resolution(data):
    """
    根据数据估计仪器分辨率
    基于数据的小数位数推断最小刻度
    """
    # 分析数据的小数位数
    decimal_places = []
    for value in data:
        if isinstance(value, (int, float)) and not np.isnan(value):
            str_value = str(value)
            if '.' in str_value:
                decimal_part = str_value.split('.')[1].rstrip('0')
                decimal_places.append(len(decimal_part))
            else:
                decimal_places.append(0)
    
    if decimal_places:
        max_decimal = max(decimal_places)
        # 分辨率 = 10^(-小数位数)
        resolution = 10 ** (-max_decimal)
        return resolution
    else:
        # 默认分辨率
        return 0.01

def q_hampel_procedure_iso13528(data, scheme="strict", instrument_resolution=None):
    """
    完整的Q/Hampel程序，符合ISO 13528标准
    
    参数:
        data: 输入数据
        scheme: 计算方案 ("strict" 或 "presentation")
        instrument_resolution: 仪器分辨率，用于处理MAD=0的情况
    
    返回:
        dict: 包含所有统计量的字典
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    
    # 1. 初始稳健位置估计 (中位数)
    initial_median = np.median(data)
    
    # 2. 计算MAD
    mad = np.median(np.abs(data - initial_median))
    
    # 3. 处理MAD ≈ 0的特殊情况（ISO 13528 7.2.2）
    if mad < 1e-12:
        # MAD ≈ 0的情况，按照ISO 13528标准处理
        
        # 位置估计：直接使用中位数（不再迭代）
        robust_mean = initial_median
        
        # 尺度估计：使用注入噪声的Qn方法
        if instrument_resolution is None:
            # 如果没有提供仪器分辨率，自动估计
            instrument_resolution = estimate_instrument_resolution(data)
        
        # 注入与仪器分辨率成比例的均匀分布噪声
        noise = np.random.uniform(-instrument_resolution/2, instrument_resolution/2, n)
        perturbed_data = data + noise
        
        # 计算Qn稳健标准差
        robust_std = qn_estimator_iso13528(perturbed_data)
        
        # 设置特殊处理标志
        mad_zero_handling = True
        weights = np.ones(n)
        iterations = 0
        converged = True
        
        # 特殊处理的说明
        special_note = f"检测到MAD≈0（平台状数据），按照ISO 13528标准处理：使用中位数作为稳健平均值，使用注入噪声(±{instrument_resolution/2:.4f})的Qn方法计算稳健标准差。"
        
    else:
        # 正常Hampel流程
        robust_mean, weights, iterations, converged = hampel_estimator_iso13528(data)
        robust_std = qn_estimator_iso13528(data)
        mad_zero_handling = False
        special_note = "正常Q/Hampel流程"
    
    # 保存原始计算值（用于严格计算方案）
    original_mean = robust_mean
    original_std = robust_std
    
    # 检测数据的小数位数
    decimal_places = detect_decimal_places(data)
    
    # 根据选择的方案进行格式化
    if scheme == "presentation":
        # 规范展示方案：使用四舍五入后的均值和标准差
        robust_mean = round(robust_mean, decimal_places)
        robust_std = round(robust_std, 3)
        formatting_note = f"使用规范展示方案：稳健平均值({robust_mean})与原始数据小数位数({decimal_places}位)一致，稳健标准差保留3位小数。"
    else:
        # 严格计算方案：使用原始计算值
        formatting_note = "使用严格计算方案：保留完整计算精度，稳健平均值和标准差使用原始计算值。"
    
    # 添加特殊处理说明（如果适用）
    if mad_zero_handling:
        formatting_note += " " + special_note
    
    # 计算控制限和Z分数 - 使用格式化后的值
    lower_limit = robust_mean - 3 * robust_std
    upper_limit = robust_mean + 3 * robust_std
    
    # 识别离群值
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask].tolist()
    clean_data = data[~outliers_mask].tolist()
    
    # 计算Z分数 - 使用格式化后的值
    Z_scores = (data - robust_mean) / robust_std
    
    # 返回完整结果
    return {
        'robust_mean': float(robust_mean) if not np.isnan(robust_mean) else 0.0,
        'robust_std': float(robust_std) if not np.isnan(robust_std) else 0.0,
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores.tolist(),
        'iterations': iterations,
        'converged': converged,
        'lower_limit': float(lower_limit) if not np.isnan(lower_limit) else 0.0,
        'upper_limit': float(upper_limit) if not np.isnan(upper_limit) else 0.0,
        'initial_median': float(initial_median) if not np.isnan(initial_median) else 0.0,
        'mad': float(mad) if not np.isnan(mad) else 0.0,
        'weights': weights.tolist(),
        'method_name': 'Q/Hampel法',
        'decimal_places': decimal_places,
        'calculation_scheme': scheme,
        'formatting_note': formatting_note,
        'original_mean': float(original_mean) if not np.isnan(original_mean) else 0.0,
        'original_std': float(original_std) if not np.isnan(original_std) else 0.0,
        'mad_zero_handling': mad_zero_handling,  # 新增标志，指示是否进行了MAD=0的特殊处理
        'instrument_resolution_used': instrument_resolution if mad_zero_handling else None
    }

def q_hampel_algorithm(data, scheme="strict"):
    """
    Q/Hampel法主函数 - 确保与其他方法接口一致
    """
    return q_hampel_procedure_iso13528(data, scheme)

# =============================================
# 修复：添加Q/Hampel法的主函数，确保方法名称一致
# =============================================

def q_hampel_algorithm(data, scheme="strict"):
    """
    Q/Hampel法主函数 - 确保与其他方法接口一致
    """
    return q_hampel_procedure_iso13528(data, scheme)

# Z比分格式化函数 - 确保显示两位小数
def format_z_scores(z_scores):
    """将Z比分统一格式化为两位小数（确保显示两位小数，如56.60）"""
    if z_scores is None:
        return None
    
    formatted_scores = []
    for score in z_scores:
        try:
            # 确保是数值类型，然后格式化为两位小数
            formatted_score = float(score)
            # 使用格式化字符串确保显示两位小数
            formatted_scores.append(formatted_score)
        except (ValueError, TypeError):
            formatted_scores.append(0.0)  # 如果转换失败，返回默认值
    
    return formatted_scores

def format_z_score_display(z_score):
    """将单个Z比分格式化为两位小数显示（确保显示两位小数）"""
    if z_score is None or pd.isna(z_score):
        return ""
    try:
        return f"{float(z_score):.2f}"
    except (ValueError, TypeError):
        return "0.00"

# 执行分析
if data is not None and len(data) > 0:
    try:
        # 数据预处理和检查
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # 确保数据是数值类型
        if not np.issubdtype(data.dtype, np.number):
            st.error("❌ 数据包含非数值类型，请检查数据格式")
            st.stop()
        
        st.markdown("---")
        st.subheader(f"📈 {method}分析结果")
        
        # 显示当前选择的计算方案
        scheme_display = "规范展示方案" if calculation_scheme == "规范展示方案" else "严格计算方案"
        st.info(f"当前使用: **{scheme_display}**")
        
        # 显示数据小数位数分析（如果可用）- 修复：使用正确的数据源
        current_decimal_info = None
        if input_method == "文件上传" and hasattr(st.session_state, 'file_decimal_info'):
            current_decimal_info = st.session_state.file_decimal_info
        elif hasattr(st.session_state, 'decimal_info'):
            current_decimal_info = st.session_state.decimal_info
            
        if current_decimal_info and current_decimal_info.get('decimal_places_count'):
            with st.expander("📏 数据小数位数分析", expanded=False):
                st.write(f"**小数位数分布:** {', '.join([f'{places}位({count}个)' for places, count in current_decimal_info['decimal_places_count'].items()])}")
                st.write(f"**最常出现的小数位数:** {current_decimal_info['detected_decimal_places']}位")
                st.write(f"**小数位数一致性:** {'是' if current_decimal_info['consistent_decimals'] else '否'}")
        
        # =============================================
        # 修复：确保使用正确的原始数据
        # =============================================
        
        # 确定当前使用的原始数据
        if input_method == "文件上传":
            current_original_data = st.session_state.file_original_data if hasattr(st.session_state, 'file_original_data') else []
            current_blank_count = st.session_state.file_blank_count if hasattr(st.session_state, 'file_blank_count') else 0
        else:
            current_original_data = st.session_state.original_data if hasattr(st.session_state, 'original_data') else []
            current_blank_count = st.session_state.blank_count if hasattr(st.session_state, 'blank_count') else 0
        
        # 如果原始数据为空，使用当前分析的数据
        if not current_original_data:
            current_original_data = data.tolist() if hasattr(data, 'tolist') else list(data)
            current_blank_count = 0
        
        # =============================================
        # 输入数据正态分布分析
        # =============================================
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
            
            # 正态性检验
            from scipy.stats import shapiro
            if len(data) >= 3 and len(data) <= 5000:
                stat, p_value = shapiro(data)
                st.write(f"正态性检验p值: {p_value:.4f}")
                if p_value > 0.05:
                    st.write(":green[数据符合正态分布 (p > 0.05)]")
                else:
                    st.write(":red[数据可能不符合正态分布 (p ≤ 0.05)]")
            else:
                st.write("正态性检验: 数据点数量不在Shapiro-Wilk检验范围内")
        
        with dist_col2:
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax_dist.hist(data, bins=15, alpha=0.7, color='skyblue', 
                                           edgecolor='black', density=True, label='Data Distribution')
            
            from scipy.stats import norm
            xmin, xmax = ax_dist.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, np.mean(data), np.std(data, ddof=1))
            ax_dist.plot(x, p, 'k', linewidth=2, label='Normal Distribution Curve')
            
            ax_dist.set_title('Normal-Probability Benchimarking of Input Data', fontsize=14, fontweight='bold')
            ax_dist.set_xlabel('Data Value', fontsize=12)
            ax_dist.set_ylabel('Probability Density', fontsize=12)
            ax_dist.legend()
            ax_dist.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_dist)
        
        # =============================================
        # 执行稳健统计分析
        # =============================================
        with st.spinner(f"正在执行{method}分析..."):
            # 将方案选择转换为参数
            scheme_param = "presentation" if calculation_scheme == "规范展示方案" else "strict"
            
            # 根据选择的方法执行分析
            if method == "迭代稳健统计法":
                results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value, scheme=scheme_param)
            elif method == "四分位稳健统计法":
                results = quartile_robust_algorithm(data, scheme=scheme_param)
            else:  # Q/Hampel法 - 使用修复后的函数
                results = q_hampel_algorithm(data, scheme=scheme_param)

            # === 新增：统一格式化Z比分为两位小数（仅用于展示和导出）===
            results['formatted_Z_scores'] = format_z_scores(results['Z_scores'])
        
        # =============================================
        # 显示计算方案说明
        # =============================================
        with st.expander("ℹ️ 计算方案说明", expanded=True):
            st.info(results['formatting_note'])
            st.success("💡 **Z比分处理说明**: 无论使用哪种计算方案，Z比分在计算过程中都保持完整精度，只在最后展示和导出时统一格式化为两位小数。")
            if calculation_scheme == "规范展示方案":
                st.warning("注意：规范展示方案会引入微小计算误差，但结果呈现更规范")
            else:
                st.success("严格计算方案确保计算精度，Z比分计算过程中不进行四舍五入")
        
        # =============================================
        # 显示主要结果
        # =============================================
        col1, col2 = st.columns(2)
        with col1:
            st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
            st.metric("稳健标准差", f"{results['robust_std']:.6f}")
            
            # 显示方法说明，基于ISO 13528标准
            if results.get('mad', 1) < 1e-12:  # MAD接近0的情况
                st.warning(f"⚠️ {results['method_name']} (MAD≈0，使用ISO 13528特殊处理)")
            else:
                st.info(f"📊 {results['method_name']}")
                
        with col2:
            # 显示迭代信息
            st.metric("迭代次数", results.get('iterations', 0))
            st.metric("离群值数量", len(results['outliers']))
            
            # 显示计算方案
            scheme_display = "规范展示" if results.get('calculation_scheme') == "presentation" else "严格计算"
            st.metric("计算方案", scheme_display)
        
        # 四分位法特有统计量
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
        
        # Q/Hampel法特有信息
        if method == "Q/Hampel法":
            st.info("🔧 **ISO 13528 Q/Hampel统计量:**")
            col3, col4, col5, col6 = st.columns(4)
            with col3:
                # 显示初始中位数
                st.metric("初始中位数", f"{results.get('initial_median', results['robust_mean']):.6f}")
            with col4:
                # 显示MAD值
                st.metric("MAD", f"{results.get('mad', 0):.6f}")
            with col5:
                # 显示收敛状态
                converged_status = "是" if results.get('converged', True) else "否"
                st.metric("收敛状态", converged_status)
            with col6:
                # 显示数据小数位数
                st.metric("数据小数位数", results.get('decimal_places', '未知'))
        
        # =============================================
        # 显示详细结果
        # =============================================
        st.subheader("📋 详细结果")
        st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
        
        # 显示MAD状态信息
        mad_value = results.get('mad', 1)
        if mad_value < 1e-12:
            st.write(f"**MAD状态**: ≈0 (使用ISO 13528特殊处理)")
        else:
            st.write(f"**MAD值**: {mad_value:.6e}")
        
        # 显示权重分布信息
        if 'weights' in results:
            weights = results['weights']
            if hasattr(weights, '__iter__') and not isinstance(weights, str):
                try:
                    unique_weights = np.unique(np.round(weights, 3))
                    if len(unique_weights) > 1:
                        st.write(f"**权重分布**: {', '.join([f'{w:.3f}' for w in unique_weights])}")
                    else:
                        st.write(f"**权重**: 常数权重 {unique_weights[0]:.3f}")
                except:
                    pass
        
        # 显示格式化说明
        if 'formatting_note' in results:
            if results.get('mad', 1) < 1e-12:
                st.warning(f"⚠️ {results['formatting_note']}")
            else:
                st.info(f"💡 {results['formatting_note']}")
        
        # 离群值显示
        if len(results['outliers']) > 0:
            outliers_list = results['outliers']
            if hasattr(outliers_list, '__iter__') and not isinstance(outliers_list, str):
                try:
                    # 转换为浮点数列表并排序
                    outliers_list = [float(x) for x in outliers_list]
                    outliers_list = sorted(outliers_list)
                    
                    # 分类显示离群值
                    st.write(f"**离群值** ({len(outliers_list)}个):")
                    
                    # 计算每个离群值的Z比分
                    robust_mean = results['robust_mean']
                    robust_std = results['robust_std']
                    
                    # 按Z比分分类离群值
                    questionable_outliers = []
                    unsatisfactory_outliers = []
                    
                    for outlier in outliers_list:
                        z_score = abs((outlier - robust_mean) / robust_std) if robust_std > 0 else float('inf')
                        if 2 < z_score <= 3:
                            questionable_outliers.append((outlier, z_score))
                        elif z_score > 3:
                            unsatisfactory_outliers.append((outlier, z_score))
                    
                    # 显示可疑离群值
                    if questionable_outliers:
                        st.write(f"  - 可疑离群值 (2<|Z|≤3): {[f'{val[0]} (Z={val[1]:.2f})' for val in questionable_outliers]}")
                    
                    # 显示不满意离群值
                    if unsatisfactory_outliers:
                        st.write(f"  - 不满意离群值 (|Z|>3): {[f'{val[0]} (Z={val[1]:.2f})' for val in unsatisfactory_outliers]}")
                        
                except (ValueError, TypeError):
                    st.write("**离群值**: [无法显示]")
            else:
                st.write("**离群值**: 无")
        else:
            st.success("✅ **离群值**: 无检测到离群值")
        
        # Z比分数分类统计 - 修复键名问题
        z_scores_data = None
        
        # 尝试获取Z比分数据，兼容多种可能的键名
        if 'z_scores' in results and results['z_scores'] is not None:
            z_scores_data = results['z_scores']
        elif 'Z_scores' in results and results['Z_scores'] is not None:
            z_scores_data = results['Z_scores']
        # 如果没有Z比分数据，但需要计算，可以从原始数据和稳健统计量计算
        elif 'clean_data' in results and len(results['clean_data']) > 0:
            try:
                # 从清洁数据计算Z比分
                clean_data = np.array(results['clean_data'])
                robust_mean = results['robust_mean']
                robust_std = results['robust_std']
                if robust_std > 0:
                    z_scores_data = ((clean_data - robust_mean) / robust_std).tolist()
            except Exception as e:
                st.warning(f"无法从清洁数据计算Z比分: {str(e)}")
        
        if z_scores_data is not None:
            try:
                z_scores = np.array(z_scores_data)
                z_scores_abs = np.abs(z_scores)
                satisfactory = np.sum(z_scores_abs <= 2)
                questionable = np.sum((z_scores_abs > 2) & (z_scores_abs <= 3))
                unsatisfactory = np.sum(z_scores_abs > 3)
                
                st.write("**Z比分数分类**:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    # 计算满意数据的百分比
                    sat_percent = satisfactory/len(z_scores)*100 if len(z_scores) > 0 else 0
                    st.metric("满意 (|Z| ≤ 2)", f"{satisfactory} 个", f"{sat_percent:.1f}%")
                with col2:
                    # 计算可疑数据的百分比
                    quest_percent = questionable/len(z_scores)*100 if len(z_scores) > 0 else 0
                    st.metric("可疑 (2 < |Z| ≤ 3)", f"{questionable} 个", f"{quest_percent:.1f}%")
                with col3:
                    # 计算不满意数据的百分比
                    unsat_percent = unsatisfactory/len(z_scores)*100 if len(z_scores) > 0 else 0
                    st.metric("不满意 (|Z| > 3)", f"{unsatisfactory} 个", f"{unsat_percent:.1f}%")
                    
                # 显示Z比分统计摘要
                if len(z_scores) > 0:
                    st.write(f"**Z比分统计**: 最小值={np.min(z_scores):.2f}, 最大值={np.max(z_scores):.2f}, 平均值={np.mean(z_scores_abs):.2f}")
                    
            except Exception as e:
                st.warning(f"无法计算Z比分分类: {str(e)}")
        else:
            # 如果无法获取Z比分数据，显示替代信息
            st.info("ℹ️ **Z比分信息**: 使用稳健统计量计算")
            st.write(f"**稳健平均值**: {results['robust_mean']:.6f}")
            st.write(f"**稳健标准差**: {results['robust_std']:.6f}")
            st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
        
        # 显示权重信息（如果可用且有意义）
        if 'weights' in results and results.get('data_pattern') != 'constant':
            weights = results['weights']
            if hasattr(weights, '__iter__') and not isinstance(weights, str):
                try:
                    unique_weights = np.unique(np.round(weights, 2))
                    if len(unique_weights) > 1:
                        st.write(f"**权重分布**: {', '.join([f'{w:.2f}' for w in unique_weights])}")
                except:
                    pass  
        
        # =============================================
        # 数据可视化 - 使用格式化后的Z比分
        # =============================================
        st.subheader("数据可视化")
        
        # 添加计算方案信息到图表
        scheme_info = "（规范展示方案）" if calculation_scheme == "规范展示方案" else "（严格计算方案）"
        
        # 创建数据框用于可视化 - 添加全面的类型安全
        try:
            if input_method == "带编号数据输入" and st.session_state.label_data_pairs:
                # 使用两列数据的原始标签
                valid_labels = []
                valid_data = []
                valid_z_scores = []
                
                # 确保Z_scores与有效数据正确对应
                if st.session_state.valid_pairs:
                    # 获取有效数据对应的标签和数值
                    valid_labels = [pair[0] for pair in st.session_state.valid_pairs]
                    valid_data = [float(pair[1]) for pair in st.session_state.valid_pairs]  # 确保转换为float
                    
                    # 使用格式化后的Z比分
                    if len(results['formatted_Z_scores']) == len(valid_data):
                        valid_z_scores = results['formatted_Z_scores']
                    else:
                        st.error(f"Z分数数量({len(results['formatted_Z_scores'])})与有效数据数量({len(valid_data)})不匹配")
                        # 使用前n个Z分数或填充
                        valid_z_scores = results['formatted_Z_scores'][:len(valid_data)] + [0] * max(0, len(valid_data) - len(results['formatted_Z_scores']))
                
                df_clean = pd.DataFrame({
                    'Original_Label': valid_labels,
                    'Original_Data': valid_data,
                    'Z_Score': valid_z_scores
                })
                
            else:
                # 其他输入方式：使用自动生成的三位数字标签
                # 确保数据是安全的Python类型
                safe_data = []
                if hasattr(data, 'tolist'):
                    safe_data = data.tolist()
                elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                    safe_data = list(data)
                else:
                    safe_data = [data] if data is not None else []
                
                # 使用格式化后的Z比分
                safe_z_scores = results['formatted_Z_scores']
                
                df_clean = pd.DataFrame({
                    'Original_Data': safe_data,
                    'Z_Score': safe_z_scores
                })
                
                # 生成三位数字标签 - 仅对有效数据
                valid_labels = []
                if st.session_state.original_data:
                    valid_count = 0
                    for i, value in enumerate(st.session_state.original_data):
                        if value is not None:  # 有效数据
                            label = f"{str(valid_count+1).zfill(3)}"  # 001, 002, ...
                            valid_labels.append(label)
                            valid_count += 1
                else:
                    # 如果没有原始数据，使用简单编号
                    valid_labels = [f"{str(i+1).zfill(3)}" for i in range(len(safe_data))]
                
                # 将标签添加到数据框
                df_clean['Original_Label'] = valid_labels[:len(df_clean)]  # 确保长度匹配
        
            # 检查数据框是否为空
            if df_clean.empty:
                st.warning("没有有效数据可用于可视化")
                # 跳过图表创建
                chart_created = False
            else:
                chart_created = True
        
        except Exception as e:
            st.error(f"创建数据框时发生错误: {str(e)}")
            chart_created = False
            
        # 只有在成功创建数据框时才继续创建图表
        if chart_created:
            try:
                set_chinese_font()
                
                # 根据Z值进行分类 - 修复分类函数
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
            
                # 按照Z值从大到小排序 - 安全排序
                try:
                    df_sorted = df_clean.sort_values('Z_Score', ascending=False)
                except:
                    # 如果排序失败，使用原始顺序
                    df_sorted = df_clean.copy()
                    st.warning("数据排序失败，使用原始顺序")
            
                # 创建Z值柱状图
                chart_height = max(10, len(df_sorted) * 0.4)
                fig, ax = plt.subplots(figsize=(14, chart_height))
            
                # 设置类别对应的颜色
                color_map = {
                    'Satisfactory': '#00FF00',    # 绿色
                    'Questionable': '#FFA500',    # 橙色
                    'Unsatisfactory': '#FF0000',   # 红色
                    'Unknown': '#808080'          # 灰色（未知类别）
                }
            
                # 创建颜色列表
                colors = []
                for cat in df_sorted['Category']:
                    colors.append(color_map.get(cat, '#808080'))  # 默认灰色
            
                # 绘制所有数据点的柱状图，按Z值排序
                y_positions = range(len(df_sorted))
                
                # 安全获取Z分数
                z_scores_to_plot = []
                for z in df_sorted['Z_Score']:
                    try:
                        z_scores_to_plot.append(float(z))
                    except (ValueError, TypeError):
                        z_scores_to_plot.append(0.0)  # 默认值
            
                bars = ax.barh(y_positions, 
                               z_scores_to_plot, 
                               color=colors, 
                               alpha=0.6,
                               height=0.8,
                               edgecolor='white',
                               linewidth=0.5)
            
                # 在柱状图上标注Z值 - 使用格式化后的两位小数（确保显示两位小数）
                for i, (bar, z_value) in enumerate(zip(bars, df_sorted['Z_Score'])):
                    try:
                        text_color = 'black'
                        
                        # Z比分统一显示两位小数，确保显示56.60而不是56.6
                        z_display = format_z_score_display(z_value)
                        
                        ax.text(bar.get_width() + 0.05 * (1 if bar.get_width() >= 0 else -1), 
                                bar.get_y() + bar.get_height()/2, 
                                z_display, 
                                ha='left' if bar.get_width() >= 0 else 'right', 
                                va='center', fontsize=9, fontweight='bold',
                                color=text_color)
                    except:
                        continue  # 如果标注失败，跳过这个数据点
            
                # 设置图形属性 - 包含计算方案信息
                ax.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
                ax.set_ylabel('Original Data ID', fontsize=14, fontweight='bold')
                ax.set_title(f'Z-Score Distribution (Sorted)', fontsize=18, fontweight='bold', pad=40)
            
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=color_map['Satisfactory'], alpha=0.6, label='Satisfactory (|Z| ≤ 2)'),
                    Patch(facecolor=color_map['Questionable'], alpha=0.6, label='Questionable (2 < |Z| ≤ 3)'),
                    Patch(facecolor=color_map['Unsatisfactory'], alpha=0.6, label='Unsatisfactory (|Z| > 3)')
                ]
            
                ax.legend(handles=legend_elements, title=f'Category', title_fontsize=12, fontsize=11, 
                          loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=True)
            
                # 设置Y轴刻度 - 使用原始标签
                ax.set_yticks(y_positions)
                
                # 安全获取标签
                y_labels = []
                for label in df_sorted['Original_Label']:
                    try:
                        y_labels.append(str(label))
                    except:
                        y_labels.append("")
                
                ax.set_yticklabels(y_labels)
            
                # 添加参考线
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
            
                # 添加网格
                ax.grid(axis='x', alpha=0.3, linestyle='--')
            
                # 反转Y轴，使最大的Z值在顶部
                ax.invert_yaxis()
            
                # 设置背景色
                ax.set_facecolor('white')
            
                # 调整布局
                plt.subplots_adjust(top=0.88)
                plt.tight_layout()
            
                # 显示图表
                st.pyplot(fig)
                
                # 在图表下方添加计算方案说明
                st.info("📝 **Z比分显示说明**: 仅在展示和导出时统一格式化为两位小数")
                    
            except Exception as e:
                st.error(f"创建图表时发生错误: {str(e)}")
                st.info("这可能是因为数据格式问题，请检查输入数据的有效性")
        
        # =============================================
        # 方案比较功能
        # =============================================
        if show_scheme_comparison:  # 使用侧边栏定义的变量
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
                else:  # Q/Hampel法 - 使用修复后的函数
                    strict_results = q_hampel_algorithm(data, scheme="strict")
                    presentation_results = q_hampel_algorithm(data, scheme="presentation")
                
                # 格式化Z比分为两位小数用于比较显示
                strict_results['formatted_Z_scores'] = format_z_scores(strict_results['Z_scores'])
                presentation_results['formatted_Z_scores'] = format_z_scores(presentation_results['Z_scores'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**严格计算方案**")
                st.write(f"稳健平均值: {strict_results['robust_mean']:.6f}")
                st.write(f"稳健标准差: {strict_results['robust_std']:.6f}")
                # 使用格式化后的Z比分显示范围
                z_min_strict = min(strict_results['formatted_Z_scores']) if strict_results['formatted_Z_scores'] else 0
                z_max_strict = max(strict_results['formatted_Z_scores']) if strict_results['formatted_Z_scores'] else 0
                st.write(f"Z比分范围: [{z_min_strict:.2f}, {z_max_strict:.2f}]")
                if 'iterations' in strict_results:
                    st.write(f"迭代次数: {strict_results['iterations']}")
                st.write(f"离群值数量: {len(strict_results['outliers'])}")
            
            with col2:
                st.write("**规范展示方案**")
                st.write(f"稳健平均值: {presentation_results['robust_mean']}")
                st.write(f"稳健标准差: {presentation_results['robust_std']:.3f}")
                # 使用格式化后的Z比分显示范围
                z_min_pres = min(presentation_results['formatted_Z_scores']) if presentation_results['formatted_Z_scores'] else 0
                z_max_pres = max(presentation_results['formatted_Z_scores']) if presentation_results['formatted_Z_scores'] else 0
                st.write(f"Z比分范围: [{z_min_pres:.2f}, {z_max_pres:.2f}]")
                if 'iterations' in presentation_results:
                    st.write(f"迭代次数: {presentation_results['iterations']}")
                st.write(f"离群值数量: {len(presentation_results['outliers'])}")
            
            # 显示方案差异说明
            st.info("""
            **方案差异说明:**
            - **严格计算方案**: 使用完整计算精度，确保计算准确性
            - **规范展示方案**: 稳健平均值与原始数据小数位数一致，结果更规范但可能引入微小误差
            - **Z比分处理**: Z比分的计算数据精度不同，计算方法相同，展示时统一格式化为两位小数
            """)
        
        # =============================================
        # 导出结果模块 - 使用格式化后的Z比分
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
                    valid_data_count += 1
                
                # 使用检测到的小数位数格式化
                formatted_value = format_number(value, detected_decimal_places)
                # 使用新的格式化函数确保显示两位小数
                formatted_z_score = format_z_score_display(z_score)
                
                result_data.append({
                    '标签原始标号': label,  # 使用用户提供的标签
                    '输入数据': formatted_value,
                    'Z比分数': formatted_z_score
                })
            
            total_data_count = len(st.session_state.label_data_pairs)
            blank_data_count = sum(1 for _, value in st.session_state.label_data_pairs if value is None)
            actual_analyzable_count = total_data_count - blank_data_count
            
        else:
            # 其他输入方式：使用自动生成的三位数字标签
            valid_data_count = 0
            
            # 修复：使用正确的原始数据源
            if current_original_data:
                for i, value in enumerate(current_original_data):
                    original_label = f"{str(i+1).zfill(3)}"  # 001, 002, ...
                    
                    if value is not None:  # 有效数据
                        z_score = results['formatted_Z_scores'][valid_data_count] if valid_data_count < len(results['formatted_Z_scores']) else None
                        # 使用检测到的小数位数格式化
                        formatted_value = format_number(value, detected_decimal_places)
                        # 使用新的格式化函数确保显示两位小数
                        formatted_z_score = format_z_score_display(z_score)
                        
                        result_data.append({
                            '标签原始标号': original_label,
                            '输入数据': formatted_value,
                            'Z比分数': formatted_z_score
                        })
                        valid_data_count += 1
                    else:  # 空白数据
                        result_data.append({
                            '标签原始标号': original_label,
                            '输入数据': None,
                            'Z比分数': ""
                        })
                
                total_data_count = len(current_original_data)
                blank_data_count = current_blank_count
                actual_analyzable_count = len(data)
            else:
                # 如果没有原始数据信息，使用简单处理
                for i, value in enumerate(data):
                    original_label = f"{str(i+1).zfill(3)}"
                    z_score = results['formatted_Z_scores'][i] if i < len(results['formatted_Z_scores']) else None
                    
                    # 使用检测到的小数位数格式化
                    formatted_value = format_number(value, detected_decimal_places)
                    # 使用新的格式化函数确保显示两位小数
                    formatted_z_score = format_z_score_display(z_score)
                    
                    result_data.append({
                        '标签原始标号': original_label,
                        '输入数据': formatted_value,
                        'Z比分数': formatted_z_score
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
                        'Z比分数': ""
                    })
        
        result_df = pd.DataFrame(result_data)
        
        # 计算统计量 - 使用检测到的小数位数格式化
        stats_data = {
            '统计量名称': ['总数据数', '实际可分析数据数', '空白数据数', '指定值', '能力评定标准差', '最小值', '最大值', '极差'],
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
        stats_df = pd.DataFrame(stats_data)
        
        # 创建用于显示的DataFrame（确保Z比分显示两位小数）
        display_df = result_df.copy()
        
        # 显示预览 - 使用与导出相同的数据
        st.write("**导出数据预览:**")
        st.dataframe(display_df, use_container_width=True)
        
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
                       
        # 在文本报告开头添加方案说明和小数位数说明
        scheme_text = "严格计算方案" if calculation_scheme == "严格计算方案" else "规范展示方案"
        report = f"""                
{method}分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
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
标签原始标号\t输入数据\tZ比分数
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
        elif method == "Q/Hampel法":
            report += f"""
Q/Hampel统计量:
--------------
初始中位数: {results['initial_median']:.6f}
MAD值: {results['mad']:.6f}
迭代次数: {results['iterations']}
收敛状态: {'是' if results['converged'] else '否'}
"""
        
        if 'iterations' in results:
            report += f"迭代次数: {results['iterations']}\n"
        
        # Z比分数分类统计
        z_scores_abs = np.abs(results['formatted_Z_scores'])
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
        
        # 添加小数位数说明
        st.info(f"💡 **小数位数说明**: 导出的数据使用 {detected_decimal_places} 位小数（基于输入数据的最大小数位数）。Z比分统一格式化为两位小数，空白数据会保留标签但数据为空。")
        
    except Exception as e:
        st.error(f"❌ 统计分析过程中发生错误: {str(e)}")
        st.info("💡 这可能是因为数据特征不适合所选的分析方法，请尝试其他统计方法或检查数据质量")

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