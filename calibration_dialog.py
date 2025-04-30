import re
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QLabel, QPushButton, QLineEdit,
                             QHBoxLayout, QFileDialog, QTabWidget, QWidget,
                             QGroupBox, QPlainTextEdit)
from PyQt5.QtCore import Qt
import numpy as np
import ast
import os


class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机标定")
        self.setFixedSize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 创建选项卡
        self.tab_widget = QTabWidget()

        # 1. 自动标定标签页
        auto_tab = QWidget()
        self.setup_auto_tab(auto_tab)
        self.tab_widget.addTab(auto_tab, "自动标定")

        # 2. 手动输入标签页
        manual_tab = QWidget()
        self.setup_manual_tab(manual_tab)
        self.tab_widget.addTab(manual_tab, "手动输入")

        layout.addWidget(self.tab_widget)

        # 状态标签和按钮
        self.status_label = QLabel("准备标定...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px;")
        self.status_label.setWordWrap(True)

        calibrate_btn = QPushButton("应用标定参数")
        calibrate_btn.setStyleSheet("font-size: 14px; height: 30px;")
        calibrate_btn.clicked.connect(self.validate_and_calibrate)

        layout.addWidget(self.status_label)
        layout.addWidget(calibrate_btn)

        self.setLayout(layout)

    def setup_auto_tab(self, tab):
        layout = QVBoxLayout()
        form = QFormLayout()

        # 图像目录选择
        self.left_dir_edit = QLineEdit()
        self.right_dir_edit = QLineEdit()

        form.addRow("左相机图像目录:", self.create_browse_row(self.left_dir_edit))
        form.addRow("右相机图像目录:", self.create_browse_row(self.right_dir_edit))

        # 棋盘格参数
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 20)
        self.rows_spin.setValue(9)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 20)
        self.cols_spin.setValue(6)

        self.square_size = QDoubleSpinBox()
        self.square_size.setRange(1, 100)
        self.square_size.setValue(25.0)
        self.square_size.setSuffix(" mm")

        form.addRow("棋盘格行数(内角点):", self.rows_spin)
        form.addRow("棋盘格列数(内角点):", self.cols_spin)
        form.addRow("方格实际尺寸:", self.square_size)

        layout.addLayout(form)
        tab.setLayout(layout)

    def setup_manual_tab(self, tab):
        layout = QVBoxLayout()

        # 模板输入区域
        template_group = QGroupBox("模板化输入")
        template_layout = QVBoxLayout()

        self.template_edit = QPlainTextEdit()
        self.template_edit.setPlaceholderText("粘贴Python格式的标定参数，例如：\n"
                                              "left_camera_matrix = np.array([...])\n"
                                              "right_camera_matrix = np.array([...])\n"
                                              "left_distortion = np.array([...])\n"
                                              "right_distortion = np.array([...])\n"
                                              "R = np.array([...])\n"
                                              "T = np.array([...])")
        self.template_edit.setFixedHeight(150)

        load_template_btn = QPushButton("解析并填充参数")
        load_template_btn.clicked.connect(self.parse_template)

        template_layout.addWidget(self.template_edit)
        template_layout.addWidget(load_template_btn)
        template_group.setLayout(template_layout)

        # 参数显示区域
        param_group = QGroupBox("标定参数")
        param_layout = QFormLayout()

        # 左相机矩阵
        self.left_matrix_edit = QPlainTextEdit()
        self.left_matrix_edit.setFixedHeight(80)
        param_layout.addRow("左相机矩阵:", self.left_matrix_edit)

        # 右相机矩阵
        self.right_matrix_edit = QPlainTextEdit()
        self.right_matrix_edit.setFixedHeight(80)
        param_layout.addRow("右相机矩阵:", self.right_matrix_edit)

        # 左畸变参数
        self.left_dist_edit = QLineEdit()
        param_layout.addRow("左畸变参数:", self.left_dist_edit)

        # 右畸变参数
        self.right_dist_edit = QLineEdit()
        param_layout.addRow("右畸变参数:", self.right_dist_edit)

        # 旋转矩阵R
        self.r_matrix_edit = QPlainTextEdit()
        self.r_matrix_edit.setFixedHeight(80)
        param_layout.addRow("旋转矩阵R:", self.r_matrix_edit)

        # 平移向量T
        self.t_vector_edit = QLineEdit()
        param_layout.addRow("平移向量T:", self.t_vector_edit)

        param_group.setLayout(param_layout)

        layout.addWidget(template_group)
        layout.addWidget(param_group)
        tab.setLayout(layout)

    def create_browse_row(self, line_edit):
        row = QHBoxLayout()
        row.addWidget(line_edit)
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self.browse_directory(line_edit))
        row.addWidget(btn)
        return row

    def browse_directory(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if path:
            line_edit.setText(path)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            self.status_label.setText(f"找到 {count} 张图像在: {os.path.basename(path)}")

    def safe_parse(self, text):
        """安全解析Python字面量"""
        try:
            return ast.literal_eval(text.strip())
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"解析错误: {str(e)}")

    def parse_template(self):
        """解析模板文本并填充到对应字段"""
        try:
            template_text = self.template_edit.toPlainText().strip()
            if not template_text:
                raise ValueError("模板内容为空")

            # 创建安全的执行环境
            safe_dict = {
                'np': np,
                'array': np.array,
                'left_camera_matrix': None,
                'right_camera_matrix': None,
                'left_distortion': None,
                'right_distortion': None,
                'R': None,
                'T': None
            }

            # 验证模板文本
            forbidden = ['import', 'exec', 'eval', 'open', 'os.', 'sys.', 'subprocess.']
            for word in forbidden:
                if word in template_text.lower():
                    raise ValueError(f"模板包含禁止的操作: {word}")

            try:
                # 先进行语法检查
                ast.parse(template_text)

                # 执行模板
                exec(template_text, {'__builtins__': None, 'np': np}, safe_dict)
            except SyntaxError as e:
                raise ValueError(f"模板语法错误: {str(e)}")
            except Exception as e:
                raise ValueError(f"模板执行错误: {str(e)}")

            # 提取并验证参数
            required_params = {
                'left_camera_matrix': (3, 3),
                'right_camera_matrix': (3, 3),
                'left_distortion': (5,),
                'right_distortion': (5,),
                'R': (3, 3),
                'T': (3,)
            }

            extracted = {}
            for name, shape in required_params.items():
                value = safe_dict.get(name)
                if value is None:
                    raise ValueError(f"缺少参数: {name}")

                try:
                    arr = np.array(value, dtype=np.float64)
                    if arr.shape != shape:
                        raise ValueError(
                            f"参数 {name} 形状应为 {shape}，实际为 {arr.shape}"
                        )
                    extracted[name] = arr
                except Exception as e:
                    raise ValueError(f"参数 {name} 格式错误: {str(e)}")

            # 填充到UI控件
            self.left_matrix_edit.setPlainText(self._format_matrix(extracted['left_camera_matrix']))
            self.right_matrix_edit.setPlainText(self._format_matrix(extracted['right_camera_matrix']))
            self.left_dist_edit.setText(self._format_array(extracted['left_distortion']))
            self.right_dist_edit.setText(self._format_array(extracted['right_distortion']))
            self.r_matrix_edit.setPlainText(self._format_matrix(extracted['R']))
            self.t_vector_edit.setText(self._format_array(extracted['T']))

            self.status_label.setStyleSheet("color: green;")
            self.status_label.setText("模板解析成功！参数已填充")

        except Exception as e:
            self.status_label.setStyleSheet("color: red;")
            self.status_label.setText(f"模板解析错误: {str(e)}")

    def _format_matrix(self, matrix):
        """格式化矩阵为可读字符串"""
        return "[\n" + ",\n".join(["    [" + ", ".join(f"{x:.6f}" for x in row) + "]"
                                   for row in np.array(matrix)]) + "\n]"

    def _format_array(self, array):
        """格式化数组为可读字符串"""
        return "[" + ", ".join(f"{x:.6f}" for x in np.array(array).flatten()) + "]"

    def preprocess_input(self, text):
        """预处理输入文本"""
        # 1. 标准化换行和空格
        text = ' '.join(text.split())  # 合并所有空白字符为单个空格

        # 2. 处理中文标点和特殊格式
        replacements = {
            '，': ',',
            '；': ';',
            '【': '[',
            '】': ']',
            '（': '(',
            '）': ')',
            '“': '"',
            '”': '"'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 3. 移除不可见字符但保留必要的结构字符
        keep_chars = set('0123456789.,-[] \t\n')
        text = ''.join(c for c in text if c in keep_chars)

        return text.strip()
    def validate_and_calibrate(self):
        if self.tab_widget.currentIndex() == 0:  # 自动标定模式
            left_dir = self.left_dir_edit.text()
            right_dir = self.right_dir_edit.text()

            if not left_dir or not right_dir or not os.path.exists(left_dir) or not os.path.exists(right_dir):
                self.status_label.setStyleSheet("color: red;")
                self.status_label.setText("图像目录无效或不存在")
                return

            self.parent().start_calibration(
                left_dir,
                right_dir,
                (self.rows_spin.value(), self.cols_spin.value()),
                self.square_size.value()
            )
            self.close()
        else:  # 手动输入
            try:
                # 预处理输入
                left_matrix_text = self.preprocess_input(self.left_matrix_edit.toPlainText())
                right_matrix_text = self.preprocess_input(self.right_matrix_edit.toPlainText())
                left_dist_text = self.preprocess_input(self.left_dist_edit.text())
                right_dist_text = self.preprocess_input(self.right_dist_edit.text())
                r_matrix_text = self.preprocess_input(self.r_matrix_edit.toPlainText())
                t_vector_text = self.preprocess_input(self.t_vector_edit.text())

                # 解析参数
                left_matrix = self.parse_matrix(left_matrix_text)
                right_matrix = self.parse_matrix(right_matrix_text)
                left_dist = self.parse_array(left_dist_text)
                right_dist = self.parse_array(right_dist_text)
                R = self.parse_matrix(r_matrix_text)
                T = self.parse_array(t_vector_text)

                # 传递给主窗口
                self.parent().set_manual_calibration(
                    left_matrix, left_dist,
                    right_matrix, right_dist,
                    R, T
                )
                self.close()

            except Exception as e:
                self.status_label.setStyleSheet("color: red;")
                self.status_label.setText(f"参数错误: {str(e)}")
                # 在控制台打印详细错误
                import traceback
                traceback.print_exc()

    def parse_matrix(self, text):
        """解析文本形式的矩阵为numpy数组"""
        try:
            # 1. 清理输入：移除所有非必要字符
            cleaned = re.sub(r'[^\d\.,\-$$$$]', ' ', text)  # 保留数字、点、逗号、负号和方括号

            # 2. 提取所有数字
            numbers = []
            for token in cleaned.split():
                # 处理每个可能是数字的token
                token = token.strip('[],')  # 移除可能附着在数字上的符号
                if token:
                    try:
                        numbers.append(float(token))
                    except ValueError:
                        continue

            # 3. 验证数字数量
            if len(numbers) != 9:
                raise ValueError(f"需要9个数字，找到{len(numbers)}个")

            # 4. 构建3x3矩阵
            matrix = np.array(numbers, dtype=np.float64).reshape(3, 3)

            return matrix

        except Exception as e:
            raise ValueError(
                f"矩阵解析失败: {str(e)}\n原始输入: '{text}'\n提取到的数字: {numbers if 'numbers' in locals() else '无'}")

    def parse_array(self, text):
        """解析文本形式的向量为numpy数组"""
        try:
            # 更严格的清理和验证
            cleaned = re.sub(r'[^\d\.,-]', '', text)  # 只保留数字、逗号、点和负号
            if not cleaned:
                return None

            # 处理可能的中文逗号或空格分隔
            elements = []
            for num in re.split(r'[,，\s]+', cleaned):
                if num:  # 跳过空字符串
                    try:
                        # 处理可能的千位分隔符
                        num = num.replace(',', '').replace('，', '')
                        elements.append(float(num))
                    except ValueError:
                        continue

            if not elements:
                raise ValueError("未找到有效的数字")

            return np.array(elements, dtype=np.float64)

        except Exception as e:
            raise ValueError(f"向量解析失败: {str(e)}\n原始输入: '{text}'")

