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

        # 相机矩阵输入区域
        cam_matrix_group = QGroupBox("相机内参矩阵 (Python列表格式)")
        cam_matrix_layout = QHBoxLayout()

        self.left_matrix_edit = QPlainTextEdit()
        self.left_matrix_edit.setPlaceholderText("[[467.0490, 0, 340.5560],\n[0, 466.8152, 244.9560],\n[0, 0, 1]]")
        self.left_matrix_edit.setFixedHeight(120)

        self.right_matrix_edit = QPlainTextEdit()
        self.right_matrix_edit.setPlaceholderText("[[467.4363, 0, 310.0826],\n[0, 467.3342, 246.0312],\n[0, 0, 1]]")
        self.right_matrix_edit.setFixedHeight(120)

        cam_matrix_layout.addWidget(QLabel("左相机:"))
        cam_matrix_layout.addWidget(self.left_matrix_edit)
        cam_matrix_layout.addWidget(QLabel("右相机:"))
        cam_matrix_layout.addWidget(self.right_matrix_edit)
        cam_matrix_group.setLayout(cam_matrix_layout)

        # 畸变参数输入区域
        dist_group = QGroupBox("畸变参数 (Python列表格式)")
        dist_layout = QHBoxLayout()

        self.left_dist_edit = QLineEdit()
        self.left_dist_edit.setPlaceholderText("[0.0645, -0.0989, 0, 0, 0]")

        self.right_dist_edit = QLineEdit()
        self.right_dist_edit.setPlaceholderText("[0.0543, -0.0681, 0, 0, 0]")

        dist_layout.addWidget(QLabel("左相机:"))
        dist_layout.addWidget(self.left_dist_edit)
        dist_layout.addWidget(QLabel("右相机:"))
        dist_layout.addWidget(self.right_dist_edit)
        dist_group.setLayout(dist_layout)

        # 外参矩阵输入区域
        ext_group = QGroupBox("外参参数 (Python列表格式)")
        ext_layout = QVBoxLayout()

        self.r_matrix_edit = QPlainTextEdit()
        self.r_matrix_edit.setPlaceholderText(
            "[[0.9999, -0.0012, 0.0128],\n[0.0012, 0.9999, 0.0019],\n[-0.0128, -0.0019, 0.9999]]")
        self.r_matrix_edit.setFixedHeight(120)

        self.t_vector_edit = QLineEdit()
        self.t_vector_edit.setPlaceholderText("[59.3051, 0.2422, -0.8870]")

        ext_layout.addWidget(QLabel("旋转矩阵R:"))
        ext_layout.addWidget(self.r_matrix_edit)
        ext_layout.addWidget(QLabel("平移向量T:"))
        ext_layout.addWidget(self.t_vector_edit)
        ext_group.setLayout(ext_layout)

        layout.addWidget(cam_matrix_group)
        layout.addWidget(dist_group)
        layout.addWidget(ext_group)
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

        else:  # 手动输入模式
            try:
                # 解析左相机矩阵
                left_matrix = np.array(self.safe_parse(self.left_matrix_edit.toPlainText()), dtype=np.float64)

                # 解析右相机矩阵
                right_matrix = np.array(self.safe_parse(self.right_matrix_edit.toPlainText()), dtype=np.float64)

                # 解析畸变参数
                left_dist = np.array(self.safe_parse(self.left_dist_edit.text()), dtype=np.float64)
                right_dist = np.array(self.safe_parse(self.right_dist_edit.text()), dtype=np.float64)

                # 解析外参
                R = np.array(self.safe_parse(self.r_matrix_edit.toPlainText()), dtype=np.float64)
                T = np.array(self.safe_parse(self.t_vector_edit.text()), dtype=np.float64)

                # 验证矩阵形状
                if left_matrix.shape != (3, 3) or right_matrix.shape != (3, 3):
                    raise ValueError("相机矩阵必须是3x3的矩阵")
                if left_dist.shape != (5,) and left_dist.shape != (5, 1):
                    raise ValueError("左相机畸变参数必须是5个元素")
                if right_dist.shape != (5,) and right_dist.shape != (5, 1):
                    raise ValueError("右相机畸变参数必须是5个元素")
                if R.shape != (3, 3):
                    raise ValueError("旋转矩阵必须是3x3")
                if T.shape != (3,) and T.shape != (3, 1):
                    raise ValueError("平移向量必须是3维")

                # 传递给主窗口
                self.parent().set_manual_calibration(
                    left_matrix, left_dist.flatten(),
                    right_matrix, right_dist.flatten(),
                    R, T.flatten()
                )
                self.status_label.setStyleSheet("color: green;")
                self.status_label.setText("手动参数设置成功！")
                self.close()

            except Exception as e:
                self.status_label.setStyleSheet("color: red;")
                self.status_label.setText(f"参数错误: {str(e)}")

