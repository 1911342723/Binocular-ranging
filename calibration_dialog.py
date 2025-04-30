# calibration_dialog.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QLabel, QPushButton, QLineEdit, QHBoxLayout, QFileDialog
import os

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机标定")
        self.setFixedSize(500, 300)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        self.left_dir_edit = QLineEdit()
        self.right_dir_edit = QLineEdit()

        form.addRow("左相机图像目录:", self.create_browse_row(self.left_dir_edit))
        form.addRow("右相机图像目录:", self.create_browse_row(self.right_dir_edit))

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

        self.status_label = QLabel("准备标定...")
        self.status_label.setWordWrap(True)

        calibrate_btn = QPushButton("开始标定")
        calibrate_btn.clicked.connect(self.validate_and_calibrate)

        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(calibrate_btn)

        self.setLayout(layout)

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

    def validate_and_calibrate(self):
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
