import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
import re
from collections import Counter
from PIL import Image
from scipy import stats
from scipy.stats import norm
from scipy import interpolate
import matplotlib as mpl
import base64
from io import BytesIO

# =============================================
# 常量定义
# =============================================

EXAMPLE_DATA = [
    54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
    54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
    54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
    54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
    54.5, 55.9, 53.2, 54.6
]

Z_SCORE_CATEGORIES = {
    "满意": (0, 2),
    "可疑": (2, 3),
    "不满意": (3, float('inf'))
}

# =============================================
# 配置函数
# =============================================

def set_chinese_font():
    """设置中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

# =============================================
# 核心工具函数
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
# 数据验证器
# =============================================

class DataValidator:
    """统一数据验证器"""
    
    @staticmethod
    def validate_numeric_string(data_string, allow_blanks=True):
        """验证数值字符串格式"""
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
                        decimal_places = 0
                        if '.' in str_value:
                            decimal_part = str_value.split('.')[1].rstrip('0')
                            decimal_places = len(decimal_part)
                        
                        decimal_info['max_decimal_places'] = max(decimal_info['max_decimal_places'], decimal_places)
                        decimal_info['decimal_places_count'][decimal_places] = \
                            decimal_info['decimal_places_count'].get(decimal_places, 0) + 1
                        
                        if previous_decimal_places is not None and previous_decimal_places != decimal_places:
                            decimal_info['consistent_decimals'] = False
                        previous_decimal_places = decimal_places
                        
                    except ValueError:
                        return False, [], [], 0, f"第{line_num}行 '{item}' 不是有效的数字", {}
                elif allow_blanks:
                    original_data.append(None)
                    blank_count += 1
        
        decimal_info['detected_decimal_places'] = decimal_info['max_decimal_places']
        return True, original_data, clean_data, blank_count, "数据格式验证通过", decimal_info
    
    @staticmethod
    def validate_data_range_variance(data):
        """验证数据范围和方差"""
        if len(data) == 0:
            return False, "数据数组为空"
        
        if len(data) < 2:
            return False, "数据点不足，无法计算方差"
        
        if np.min(data) == np.max(data):
            return False, "所有数据值相同，无法进行统计分析"
        
        variance = np.var(data, ddof=1)
        if variance == 0:
            return False, "数据方差为零，所有数据点相同"
        
        return True, f"数据范围: [{np.min(data):.4f}, {np.max(data):.4f}], 方差: {variance:.6f}"
    
    @staticmethod
    def detect_outliers_iqr(data):
        """使用IQR方法检测异常值"""
        if len(data) < 3:
            return [], ["数据点不足，无法进行异常值检测"]
        
        try:
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [float(v) for v in data if v < lower_bound or v > upper_bound]
            
            if outliers:
                return outliers, [f"检测到 {len(outliers)} 个潜在异常值（基于IQR方法）"]
            return [], ["未发现明显异常值"]
        except Exception as e:
            return [], [f"异常值检测错误: {str(e)}"]
    
    @staticmethod
    def comprehensive_validation(data_string, calculation_scheme="严格计算方案"):
        """综合数据验证"""
        validation_report = []
        
        is_valid, original_data, clean_data, blank_count, msg, decimal_info = \
            DataValidator.validate_numeric_string(data_string)
        
        if not is_valid:
            return False, [], [], blank_count, [f"❌ {msg}"], decimal_info
        validation_report.append(f"✅ {msg}")
        
        is_valid, range_msg = DataValidator.validate_data_range_variance(clean_data)
        if not is_valid:
            return False, original_data, [], blank_count, validation_report + [f"❌ {range_msg}"], decimal_info
        validation_report.append(f"✅ {range_msg}")
        
        outliers, outliers_msg = DataValidator.detect_outliers_iqr(clean_data)
        if outliers_msg and "检测到" in outliers_msg[0]:
            validation_report.append(f"⚠️ {outliers_msg[0]}")
            if outliers:
                validation_report.append(f"   异常值: {', '.join([f'{x:.4f}' for x in outliers])}")
        else:
            validation_report.append("✅ 未发现明显异常值")
        
        if blank_count > 0:
            validation_report.append(f"⚠️ 检测到 {blank_count} 个空白数据点，这些数据将被忽略")
        else:
            validation_report.append("✅ 未发现空白数据")
        
        validation_report.extend([
            f"📈 数据统计摘要:",
            f"   总数据点数: {len(original_data)}",
            f"   实际可分析数据数: {len(clean_data)}",
            f"   有效数据范围: [{np.min(clean_data):.4f}, {np.max(clean_data):.4f}]",
            f"   有效数据平均值: {np.mean(clean_data):.4f}",
            f"   有效数据标准差: {np.std(clean_data, ddof=1):.4f}"
        ])
        
        return True, original_data, clean_data, blank_count, validation_report, decimal_info

# =============================================
# 文件处理器
# =============================================

class FileProcessor:
    """文件处理类"""
    
    @staticmethod
    def detect_format(uploaded_file):
        """自动检测文件格式"""
        filename = uploaded_file.name.lower()
        
        if filename.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith('.json'):
            return 'json'
        else:
            return 'txt'
    
    @staticmethod
    def read_file(uploaded_file, file_format):
        """读取文件为DataFrame"""
        try:
            if file_format == 'excel':
                df = pd.read_excel(uploaded_file, na_filter=True)
            elif file_format == 'csv':
                df = pd.read_csv(uploaded_file, na_filter=True)
            elif file_format == 'json':
                df = pd.read_json(uploaded_file)
            else:  # txt
                content = uploaded_file.read().decode('utf-8')
                df = pd.read_csv(io.StringIO(content), sep=None, engine='python', na_filter=True)
            
            return df, True, None
        except Exception as e:
            return None, False, str(e)
    
    @staticmethod
    def extract_numeric_data(df):
        """从DataFrame中提取数值数据"""
        if df is None or df.empty:
            return None, [], 0, {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return None, [], 0, {}
        
        selected_column = numeric_columns[0]
        data_column = df[selected_column].dropna()
        
        clean_data = data_column.values
        original_data = df[selected_column].tolist()
        blank_count = len(df[selected_column]) - len(data_column)
        
        decimal_info = {
            'decimal_places_count': {},
            'max_decimal_places': 0,
            'consistent_decimals': True,
            'detected_decimal_places': detect_decimal_places(clean_data)
        }
        
        return clean_data, original_data, blank_count, decimal_info

# =============================================
# 统一样式管理
# =============================================

class UIManager:
    """UI管理器"""
    
    @staticmethod
    def setup_page():
        """设置页面配置和样式"""
        st.set_page_config(
            page_title="统计宝 | 稳健统计分析工具",
            page_icon="📊",
            layout="wide"
        )
        
        # 添加CCS样式
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
                st.image("??", width=100)
        with col2:
            st.markdown("### **统计宝**")
            st.markdown("提供多种稳健统计分析方法，用于处理包含异常值的数据集。")   
    
    @staticmethod
    def display_sidebar(method, show_scheme_comparison):
        """显示侧边栏"""
        st.sidebar.header("⚙️ 分析设置")
        
        method = st.sidebar.selectbox(
            "选择统计方法:",
            ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法", "Z比分计算模块"]
        )
        
        if method == "迭代稳健统计法":
            st.sidebar.subheader("迭代法参数")
            k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
            max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)
        else:
            k_value = 1.5
            max_iter = 50
        
        st.sidebar.subheader("计算方案")
        calculation_scheme = st.sidebar.radio(
            "选择计算方案:",
            ["严格计算方案", "规范展示方案"]
        )
        
        show_scheme_comparison = st.sidebar.checkbox("显示方案比较", value=False)
        
        return method, calculation_scheme, k_value, max_iter, show_scheme_comparison

# =============================================
# 统计方法实现
# =============================================

class StatisticalMethods:
    """统计方法实现"""
    
    @staticmethod
    def iterative_robust(data, k=1.5, max_iter=50, scheme="strict"):
        """迭代稳健统计法"""
        n = len(data)
        X_star = np.median(data)
        S_star = 1.483 * np.median(np.abs(data - X_star))
        
        history = []
        for iteration in range(max_iter):
            delta = k * S_star
            Xj_star = np.clip(data, X_star - delta, X_star + delta)
            new_X_star = np.mean(Xj_star)
            new_S_star = 1.134 * np.sqrt(np.sum((Xj_star - new_X_star)**2) / (n-1))
            
            history.append({
                'iteration': iteration + 1,
                'X_star': new_X_star,
                'S_star': new_S_star
            })
            
            if abs(new_X_star - X_star) < 1e-6 and abs(new_S_star - S_star) < 1e-6:
                break
                
            X_star, S_star = new_X_star, new_S_star
        
        return StatisticalMethods._format_results(
            data, X_star, S_star, k, scheme, "迭代稳健统计法", history
        )
    
    @staticmethod
    def quartile_robust(data, scheme="strict"):
        """四分位稳健统计法"""
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        niqr = 0.7413 * iqr
        
        return StatisticalMethods._format_results(
            data, median, niqr, 1.5, scheme, "四分位稳健统计法",
            {'q1': q1, 'q3': q3, 'iqr': iqr, 'niqr': niqr}
        )
    
    @staticmethod
    def z_score_calculation(data, robust_mean, robust_std, scheme="strict"):
        """Z比分计算"""
        if robust_std <= 0:
            raise ValueError("稳健标准差必须大于0")
        
        return StatisticalMethods._format_results(
            data, robust_mean, robust_std, 3, scheme, "Z比分计算模块"
        )
    
    @staticmethod
    def _format_results(data, robust_mean, robust_std, k, scheme, method_name, extra_info=None):
        """统一格式化结果"""
        decimal_places = detect_decimal_places(data)
        
        if scheme == "presentation":
            display_mean = round(robust_mean, decimal_places)
            display_std = round(robust_std, 3)
            formatting_note = f"规范展示方案：稳健平均值({display_mean})与原始数据小数位数一致"
        else:
            display_mean = robust_mean
            display_std = robust_std
            formatting_note = "严格计算方案：保留完整计算精度"
        
        lower_limit = display_mean - k * display_std
        upper_limit = display_mean + k * display_std
        
        outliers = data[(data < lower_limit) | (data > upper_limit)]
        clean_data = data[(data >= lower_limit) & (data <= upper_limit)]
        
        z_scores = (data - display_mean) / display_std if display_std > 0 else np.zeros_like(data)
        z_scores_rounded = np.round(z_scores, 2)
        classifications = [classify_z_score(z) for z in z_scores_rounded]
        
        result = {
            'robust_mean': float(display_mean),
            'robust_std': float(display_std),
            'clean_data': clean_data.tolist(),
            'outliers': outliers.tolist(),
            'Z_scores_rounded': z_scores_rounded.tolist(),
            'z_score_classifications': classifications,
            'lower_limit': float(lower_limit),
            'upper_limit': float(upper_limit),
            'method_name': method_name,
            'calculation_scheme': scheme,
            'formatting_note': formatting_note,
            'decimal_places': decimal_places
        }
        
        if extra_info:
            result.update(extra_info)
        
        return result

# =============================================
# 结果显示组件
# =============================================

class ResultDisplayer:
    """结果显示组件"""
    
    @staticmethod
    def display_core_metrics(results):
        """显示核心指标"""
        cols = st.columns(4)
        cols[0].metric("稳健平均值", f"{results['robust_mean']:.6f}")
        cols[1].metric("稳健标准差", f"{results['robust_std']:.6f}")
        cols[2].metric("离群值数量", len(results['outliers']))
        if 'iterations' in results:
            cols[3].metric("迭代次数", results['iterations'])
    
    @staticmethod
    def display_z_score_stats(results):
        """显示Z比分统计"""
        classifications = results['z_score_classifications']
        counts = Counter(classifications)
        total = len(classifications)
        
        cols = st.columns(3)
        for i, (category, (min_val, max_val)) in enumerate(Z_SCORE_CATEGORIES.items()):
            count = counts.get(category, 0)
            percentage = (count / total * 100) if total > 0 else 0
            cols[i].metric(
                f"{category} ({min_val}≤|Z|{'' if max_val==float('inf') else '<'+str(max_val)})",
                f"{count} 个",
                f"{percentage:.1f}%"
            )
    
    @staticmethod
    def create_z_score_chart(results, labels=None):
        """创建Z比分图表"""
        set_chinese_font()
        
        z_scores = results['Z_scores_rounded']
        classifications = results['z_score_classifications']
        
        if not z_scores:
            return None
        
        n_points = len(z_scores)
        if labels is None or len(labels) != n_points:
            labels = [f"{i+1:03d}" for i in range(n_points)]
        
        chart_data = pd.DataFrame({
            'Label': labels,
            'Z_Score': z_scores,
            'Classification': classifications
        }).sort_values('Z_Score', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, max(10, n_points * 0.4)))
        
        color_map = {'满意': '#00FF00', '可疑': '#FFA500', '不满意': '#FF0000'}
        colors = [color_map.get(cat, '#808080') for cat in chart_data['Classification']]
        
        y_pos = range(len(chart_data))
        bars = ax.barh(y_pos, chart_data['Z_Score'], color=colors, alpha=0.6, height=0.8)
        
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('数据点')
        ax.set_title('Z比分分布图')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(chart_data['Label'])
        
        ax.axvline(x=0, color='black', alpha=0.5, linewidth=1)
        for threshold in [-3, -2, 2, 3]:
            ax.axvline(x=threshold, color='red' if abs(threshold)==3 else 'gray', 
                      linestyle='--', alpha=0.7, linewidth=0.8)
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return fig

# =============================================
# 主应用
# =============================================

class StatsApp:
    """主应用类"""
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.method = "迭代稳健统计法"
        self.calculation_scheme = "严格计算方案"
        
    def run(self):
        """运行应用"""
        UIManager.setup_page()
        UIManager.display_header()
        
        self.method, self.calculation_scheme, k_value, max_iter, show_scheme_comparison = \
            UIManager.display_sidebar(self.method, False)
        
        self.data_input_section()
        
        if self.data is not None:
            self.analysis_section(k_value, max_iter, show_scheme_comparison)
    
    def data_input_section(self):
        """数据输入部分"""
        st.markdown("### **数据输入方式**")
        input_method = st.radio(
            "",
            ["手动输入", "带编号数据输入", "文件上传", "示例数据"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_method == "手动输入":
            self._manual_input()
        elif input_method == "带编号数据输入":
            self._labeled_input()
        elif input_method == "文件上传":
            self._file_upload()
        else:
            self._example_data()
    
    def _manual_input(self):
        """手动输入处理"""
        st.subheader("📝 手动输入数据")
        
        default_data = ", ".join(map(str, EXAMPLE_DATA))
        input_text = st.text_area(
            "请输入数据（用逗号、空格或换行分隔）:",
            value=default_data,
            height=150
        )
        
        if st.button("分析数据", type="primary"):
            self._process_input_data(input_text)
    
    def _labeled_input(self):
        """带标签输入处理"""
        st.subheader("📝 带编号数据输入")
        
        example = "Sample_A, 54.4\nSample_B, 54.6\nControl_1, 54.2"
        input_text = st.text_area(
            "请输入标签和数值（每行格式：标签, 数值）:",
            value=example,
            height=200
        )
        
        if st.button("分析带标签数据", type="primary"):
            self._process_labeled_data(input_text)
    
    def _file_upload(self):
        """文件上传处理"""
        st.subheader("📁 上传数据文件")
        
        uploaded_file = st.file_uploader(
            "选择数据文件 (CSV, Excel, TXT, JSON)",
            type=['csv', 'xlsx', 'xls', 'txt', 'json']
        )
        
        if uploaded_file is not None:
            file_format = FileProcessor.detect_format(uploaded_file)
            df, success, error = FileProcessor.read_file(uploaded_file, file_format)
            
            if success and df is not None:
                clean_data, original_data, blank_count, decimal_info = \
                    FileProcessor.extract_numeric_data(df)
                
                if clean_data is not None and len(clean_data) > 0:
                    self.data = clean_data
                    self.labels = None
                    
                    st.success(f"✅ 成功加载 {len(clean_data)} 个有效数据点")
                    if blank_count > 0:
                        st.warning(f"⚠️ 检测到 {blank_count} 个空白数据点")
                else:
                    st.error("❌ 无法从文件中提取有效数据")
            else:
                st.error(f"❌ 文件读取失败: {error}")
    
    def _example_data(self):
        """示例数据处理"""
        st.subheader("🎯 示例数据分析")
        
        if st.button("使用示例数据", type="primary"):
            self.data = np.array(EXAMPLE_DATA)
            self.labels = [f"示例{i+1:03d}" for i in range(len(EXAMPLE_DATA))]
            st.success(f"✅ 已加载示例数据 ({len(self.data)} 个数据点)")
    
    def _process_input_data(self, input_text):
        """处理普通输入数据"""
        is_valid, original_data, clean_data, blank_count, report, decimal_info = \
            DataValidator.comprehensive_validation(input_text, self.calculation_scheme)
        
        if is_valid:
            self.data = np.array(clean_data)
            self.labels = None
            
            st.success(f"✅ 数据验证通过！成功加载 {len(clean_data)} 个有效数据点")
            
            with st.expander("📋 查看验证报告"):
                for line in report:
                    if line.startswith("❌"):
                        st.error(line)
                    elif line.startswith("⚠️"):
                        st.warning(line)
                    else:
                        st.write(line)
        else:
            st.error("❌ 数据验证失败")
    
    def _process_labeled_data(self, input_text):
        """处理带标签数据"""
        lines = input_text.strip().split('\n')
        labels = []
        values = []
        
        for line in lines:
            parts = [p.strip() for p in re.split(r'[,;]', line) if p.strip()]
            if len(parts) >= 2:
                labels.append(parts[0])
                try:
                    values.append(float(parts[1]))
                except ValueError:
                    st.error(f"无效数值: {parts[1]}")
                    return
        
        if values:
            self.data = np.array(values)
            self.labels = labels
            st.success(f"✅ 成功加载 {len(values)} 个带标签数据点")
    
    def analysis_section(self, k_value, max_iter, show_scheme_comparison):
        """分析部分"""
        st.markdown("---")
        st.subheader(f"📈 {self.method}分析结果")
        
        scheme_param = "presentation" if self.calculation_scheme == "规范展示方案" else "strict"
        
        try:
            with st.spinner("正在执行分析..."):
                if self.method == "迭代稳健统计法":
                    results = StatisticalMethods.iterative_robust(
                        self.data, k_value, max_iter, scheme_param
                    )
                elif self.method == "四分位稳健统计法":
                    results = StatisticalMethods.quartile_robust(
                        self.data, scheme_param
                    )
                elif self.method == "Z比分计算模块":
                    col1, col2 = st.columns(2)
                    with col1:
                        robust_mean = st.number_input("稳健平均值", value=54.4, step=0.1)
                    with col2:
                        robust_std = st.number_input("稳健标准差", value=0.3, step=0.01, min_value=0.01)
                    
                    results = StatisticalMethods.z_score_calculation(
                        self.data, robust_mean, robust_std, scheme_param
                    )
                else:  # Q/Hampel法
                    results = StatisticalMethods.quartile_robust(
                        self.data, scheme_param
                    )
                    results['method_name'] = "Q/Hampel法"
            
            self._display_results(results)
            
        except Exception as e:
            st.error(f"❌ 分析过程中发生错误: {str(e)}")
    
    def _display_results(self, results):
        """显示结果"""
        st.info(results['formatting_note'])
        
        ResultDisplayer.display_core_metrics(results)
        ResultDisplayer.display_z_score_stats(results)
        
        with st.expander("📋 详细结果"):
            st.write(f"**正常值范围**: [{results['lower_limit']:.4f}, {results['upper_limit']:.4f}]")
            
            if results['outliers']:
                st.write(f"**离群值** ({len(results['outliers'])}个): {', '.join([f'{x:.4f}' for x in results['outliers']])}")
            else:
                st.success("✅ 未检测到离群值")
        
        st.subheader("📊 数据可视化")
        
        fig = ResultDisplayer.create_z_score_chart(results, self.labels)
        if fig:
            st.pyplot(fig)
        
        self._export_section(results)
    
    def _export_section(self, results):
        """导出部分"""
        st.subheader("💾 导出结果")
        
        result_df = pd.DataFrame({
            '数据点': self.labels if self.labels else [f"{i+1:03d}" for i in range(len(self.data))],
            '原始数值': self.data,
            'Z比分数': results['Z_scores_rounded'],
            '分类结果': results['z_score_classifications']
        })
        
        st.dataframe(result_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = result_df.to_csv(index=False)
            st.download_button(
                "📥 下载CSV",
                csv,
                f"{results['method_name']}_结果.csv",
                "text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            result_df.to_excel(excel_buffer, index=False)
            st.download_button(
                "📥 下载Excel",
                excel_buffer.getvalue(),
                f"{results['method_name']}_结果.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            json_data = result_df.to_json(orient='records', force_ascii=False)
            st.download_button(
                "📥 下载JSON",
                json_data,
                f"{results['method_name']}_结果.json",
                "application/json"
            )

# =============================================
# 应用入口
# =============================================

if __name__ == "__main__":
    app = StatsApp()
    app.run()
    

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