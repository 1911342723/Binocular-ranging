import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QTextEdit, QFileDialog, QDialog, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QLineEdit)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from stereo_vision_processor import StereoVisionProcessor
from calibration_dialog import CalibrationDialog

"""整体窗口的布局"""
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("双目视觉测距系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化处理器
        self.processor = StereoVisionProcessor()
        self.threeD = None

        # 设置UI
        self.setup_ui()

        # 视频捕获
        self.capture = cv2.VideoCapture("./video/car.avi")
        self.timer = QTimer()

        # 显示模式
        self.current_mode = "灰度图"

    def setup_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧布局 (原始视频)
        left_layout = QVBoxLayout()

        self.original_label = QLabel("原始视频")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid black;")
        self.original_label.setFixedSize(640, 480)
        left_layout.addWidget(self.original_label)

        # 右侧布局 (处理结果)
        right_layout = QVBoxLayout()

        # 显示模式选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["灰度图", "深度图"])
        self.mode_combo.currentTextChanged.connect(self.update_display_mode)
        right_layout.addWidget(self.mode_combo)

        # 结果显示
        self.result_label = QLabel("处理结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 1px solid black;")
        self.result_label.setFixedSize(640, 480)
        self.result_label.mousePressEvent = self.show_distance
        right_layout.addWidget(self.result_label)

        # 距离信息
        self.distance_text = QTextEdit()
        self.distance_text.setReadOnly(True)
        self.distance_text.setPlaceholderText("点击深度图显示距离信息...")
        right_layout.addWidget(self.distance_text)

        # 状态和按钮
        self.calib_status = QLabel("状态: 未标定")

        btn_layout = QHBoxLayout()
        self.calibrate_btn = QPushButton("标定相机")
        self.calibrate_btn.clicked.connect(self.show_calibration_dialog)

        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.start_video)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_video)

        btn_layout.addWidget(self.calibrate_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        right_layout.addWidget(self.calib_status)
        right_layout.addLayout(btn_layout)

        # 合并布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def show_calibration_dialog(self):
        """显示标定对话框"""
        dialog = CalibrationDialog(self)
        dialog.exec_()

    def start_calibration(self, left_dir, right_dir, chessboard_size, square_size):
        """执行相机标定"""
        try:
            ret = self.processor.calibrate_cameras(
                left_dir, right_dir,
                chessboard_size, square_size
            )

            if ret:
                self.calib_status.setText(f"状态: 已标定 (误差: {ret:.2f})")
                QMessageBox.information(self, "标定成功", f"标定完成，RMS误差: {ret:.2f}")
            else:
                self.calib_status.setText("状态: 标定失败")
                QMessageBox.warning(self, "标定失败", "请检查图像和参数设置")
        except Exception as e:
            self.calib_status.setText("状态: 标定错误")
            QMessageBox.critical(self, "标定错误", str(e))

    def start_video(self):
        """开始视频处理"""
        if not self.processor.calibrator.is_calibrated:
            QMessageBox.warning(self, "警告", "请先完成相机标定！")
            return

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms更新一帧

    def stop_video(self):
        """停止视频处理"""
        self.timer.stop()

    def update_frame(self):
        """更新视频帧"""
        ret, frame = self.capture.read()
        if ret:
            original, gray_img, depth_img, self.threeD = self.processor.process_frame(frame)

            # 显示原始视频
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            height, width, channel = original.shape
            bytes_per_line = 3 * width
            q_img = QImage(original.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.original_label.setPixmap(QPixmap.fromImage(q_img))

            # 根据模式显示结果
            if self.current_mode == "灰度图":
                display_img = gray_img
            else:
                display_img = depth_img

            height, width, channel = display_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.result_label.setPixmap(QPixmap.fromImage(q_img))

    def show_distance(self, event):
        """显示点击位置的深度信息"""
        if self.current_mode == "深度图" and self.threeD is not None:
            x = event.pos().x()
            y = event.pos().y()

            if 0 <= x < 640 and 0 <= y < 480:
                distance = np.linalg.norm(self.threeD[y][x]) / 1000  # 转换为米
                self.distance_text.clear()
                self.distance_text.append(f"点击位置: x={x}, y={y}")
                self.distance_text.append(f"距离: {distance:.2f} 米")

    def update_display_mode(self, mode):
        """更新显示模式"""
        self.current_mode = mode

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        self.capture.release()
        event.accept()
