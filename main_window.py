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
        self.current_video_path = None
        self.is_playing = False

        # 设置UI
        self.setup_ui()

        # 视频捕获
        self.capture = None
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
        self.video_path_label = QLineEdit()
        self.video_path_label.setReadOnly(True)
        self.video_path_label.setPlaceholderText("未选择视频文件")

        btn_layout = QHBoxLayout()
        self.calibrate_btn = QPushButton("标定相机")
        self.calibrate_btn.clicked.connect(self.show_calibration_dialog)

        self.select_video_btn = QPushButton("选择视频")
        self.select_video_btn.clicked.connect(self.select_video_file)

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)

        btn_layout.addWidget(self.calibrate_btn)
        btn_layout.addWidget(self.select_video_btn)
        btn_layout.addWidget(self.play_btn)

        right_layout.addWidget(self.calib_status)
        right_layout.addWidget(QLabel("当前视频:"))
        right_layout.addWidget(self.video_path_label)
        right_layout.addLayout(btn_layout)

        # 合并布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def select_video_file(self):
        """选择视频文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.avi *.mp4 *.mov *.mkv);;所有文件 (*)",
            options=options)

        if file_path:
            self.current_video_path = file_path
            self.video_path_label.setText(os.path.basename(file_path))
            self.load_video(file_path)
            self.play_btn.setEnabled(True)
            self.play_btn.setText("暂停")
            self.is_playing = True

    def load_video(self, video_path):
        """加载视频文件"""
        if self.capture is not None:
            self.capture.release()

        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件")
            return False

        # 设置定时器
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms更新一帧
        return True

    def toggle_playback(self):
        """切换播放/暂停状态"""
        if self.capture is None:
            return

        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("播放")
        else:
            self.timer.start(30)
            self.play_btn.setText("暂停")

        self.is_playing = not self.is_playing

    def show_calibration_dialog(self):
        """显示标定对话框"""
        dialog = CalibrationDialog(self)
        dialog.exec_()

    def set_manual_calibration(self, left_matrix, left_dist, right_matrix, right_dist, R, T):
        """设置手动输入的标定参数"""
        try:
            # 传递给处理器
            self.processor.calibrator.set_manual_parameters(
                left_matrix, left_dist,
                right_matrix, right_dist,
                R, T
            )

            self.calib_status.setText("状态: 已标定 (手动参数)")
            self.calib_status.setStyleSheet("color: green;")
            QMessageBox.information(self, "成功", "手动参数设置成功")

        except Exception as e:
            self.calib_status.setText("状态: 参数错误")
            self.calib_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "标定错误", str(e))

    def start_calibration(self, left_dir, right_dir, chessboard_size, square_size):
        """执行相机标定"""
        try:
            # 保存获取的标定相机的参数
            ret = self.processor.calibrate_cameras(
                left_dir, right_dir,
                chessboard_size, square_size
            )

            if ret:
                self.calib_status.setText(f"状态: 已标定 (误差: {ret:.2f})")
                self.calib_status.setStyleSheet("color: green;")
                QMessageBox.information(self, "标定成功", f"标定完成，RMS误差: {ret:.2f}")
            else:
                self.calib_status.setText("状态: 标定失败")
                self.calib_status.setStyleSheet("color: red;")
                QMessageBox.warning(self, "标定失败", "请检查图像和参数设置")
        except Exception as e:
            self.calib_status.setText("状态: 标定错误")
            self.calib_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "标定错误", str(e))

    def update_frame(self):
        """更新视频帧"""
        if self.capture is None or not self.is_playing:
            return

        ret, frame = self.capture.read()
        if not ret:
            # 视频结束，重新开始
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        try:
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

        except Exception as e:
            print(f"处理帧时出错: {str(e)}")

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
        if self.capture is not None:
            self.capture.release()
        self.timer.stop()
        event.accept()