import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QTextEdit, QFileDialog, QDialog, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QLineEdit, QStackedLayout, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
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
        self.threeD = None

    def setup_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局 - 使用网格布局更灵活
        main_layout = QGridLayout()
        central_widget.setLayout(main_layout)

        # 左侧布局 (原始视频) - 使用嵌套布局实现居中
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setAlignment(Qt.AlignCenter)  # 使内容居中

        # 原始视频标签
        self.original_label = QLabel("原始视频")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3498db;
                border-radius: 5px;
                background-color: #f8f9fa;
                padding: 5px;
            }
        """)
        self.original_label.setFixedSize(640, 480)

        left_layout.addWidget(self.original_label)
        left_layout.addStretch()

        # 右侧布局 (处理结果)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        # 显示模式选择 - 美化样式
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["灰度图", "深度图", "点云"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                min-height: 30px;
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
        """)
        self.mode_combo.currentTextChanged.connect(self.update_display_mode)
        right_layout.addWidget(self.mode_combo)

        # 结果视图容器
        self.result_container = QWidget()
        self.result_container.setFixedSize(640, 480)
        self.result_layout = QStackedLayout()
        self.result_container.setLayout(self.result_layout)

        # 美化结果视图
        self.result_label = QLabel("处理结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                border: 2px solid #2ecc71;
                border-radius: 5px;
                background-color: #f8f9fa;
                padding: 5px;
            }
        """)
        self.result_label.mousePressEvent = self.show_distance

        self.point_cloud_view = QLabel("点云显示")
        self.point_cloud_view.setAlignment(Qt.AlignCenter)
        self.point_cloud_view.setStyleSheet("""
            QLabel {
                border: 2px solid #e74c3c;
                border-radius: 5px;
                background-color: #f8f9fa;
                padding: 5px;
            }
        """)

        self.result_layout.addWidget(self.result_label)
        self.result_layout.addWidget(self.point_cloud_view)
        right_layout.addWidget(self.result_container)

        # 距离信息显示 - 美化样式
        self.distance_text = QTextEdit()
        self.distance_text.setReadOnly(True)
        self.distance_text.setAlignment(Qt.AlignCenter)  # 设置默认居中
        self.distance_text.setPlaceholderText("点击深度图显示距离信息...")
        self.distance_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #95a5a6;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                background-color: #f8f9fa;
            }
        """)
        right_layout.addWidget(self.distance_text)

        # 状态和按钮区域
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 状态标签美化
        self.calib_status = QLabel("状态: 未标定")
        self.calib_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                border-radius: 4px;
                background-color: #f39c12;
                color: white;
            }
        """)

        # 视频路径显示
        self.video_path_label = QLineEdit()
        self.video_path_label.setReadOnly(True)
        self.video_path_label.setPlaceholderText("未选择视频文件")
        self.video_path_label.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #ecf0f1;
            }
        """)

        # 按钮布局 - 美化按钮
        btn_layout = QHBoxLayout()

        self.calibrate_btn = QPushButton("标定相机")
        self.select_video_btn = QPushButton("选择视频")
        self.play_btn = QPushButton("播放")

        # 统一按钮样式
        button_style = """
            QPushButton {
                min-height: 35px;
                padding: 8px 15px;
                font-size: 14px;
                border: none;
                border-radius: 4px;
                background-color: #3498db;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """

        self.calibrate_btn.setStyleSheet(button_style)
        self.select_video_btn.setStyleSheet(button_style)
        self.play_btn.setStyleSheet(button_style)

        self.calibrate_btn.clicked.connect(self.show_calibration_dialog)
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)

        btn_layout.addWidget(self.calibrate_btn)
        btn_layout.addWidget(self.select_video_btn)
        btn_layout.addWidget(self.play_btn)

        control_layout.addWidget(self.calib_status)
        control_layout.addWidget(QLabel("当前视频:"))
        control_layout.addWidget(self.video_path_label)
        control_layout.addLayout(btn_layout)
        right_layout.addWidget(control_panel)

        # 合并布局 - 设置间距和边距
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        main_layout.addWidget(left_container, 0, 0, Qt.AlignCenter)  # 左列居中
        main_layout.addWidget(right_container, 0, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)

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
            original, gray_img, depth_img, threeD = self.processor.process_frame(frame)
            self.threeD = threeD  # 保存当前帧的三维数据
            # 显示原始视频
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            height, width, channel = original.shape
            bytes_per_line = 3 * width
            q_img = QImage(original.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.original_label.setPixmap(QPixmap.fromImage(q_img))

            # 根据模式显示结果
            if self.current_mode == "灰度图":
                display_img = gray_img
            elif self.current_mode == "深度图":
                display_img = depth_img
            else:  # 点云模式
                point_cloud_img = self.processor.generate_point_cloud(threeD)
                display_img = point_cloud_img

            # 更新当前显示的视图
            current_view = self.result_layout.currentWidget()
            height, width, channel = display_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            current_view.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")

    def show_distance(self, event):
        """显示点击位置的深度信息（优化版，解决闪烁和内存问题）"""
        try:
            if self.current_mode != "深度图" or not hasattr(self, 'threeD') or self.threeD is None:
                return

            # 获取当前显示的pixmap
            current_pixmap = self.result_label.pixmap()
            if not current_pixmap:
                return

            # 计算点击位置对应的图像坐标
            label_size = self.result_label.size()
            img_size = current_pixmap.size()
            scale_x = img_size.width() / label_size.width()
            scale_y = img_size.height() / label_size.height()

            x = int(event.pos().x() * scale_x)
            y = int(event.pos().y() * scale_y)

            # 边界检查
            if not (0 <= x < img_size.width() and 0 <= y < img_size.height()):
                return

            # 获取3D坐标信息
            point_3d = self.threeD[y][x]
            distance = np.linalg.norm(point_3d) / 1000  # 转换为米

            # 更新信息显示（居中）
            self.distance_text.clear()
            self.distance_text.setAlignment(Qt.AlignCenter)  # 设置文字居中
            self.distance_text.append("=== 点击位置信息 ===")
            self.distance_text.append(f"像素坐标: (x={x}, y={y})")
            self.distance_text.append(
                f"世界坐标: (X={point_3d[0] / 1000:.3f}m, Y={point_3d[1] / 1000:.3f}m, Z={point_3d[2] / 1000:.3f}m)")
            self.distance_text.append(f"距离相机距离: {distance:.3f} 米")

            # 创建带标记的新图像（使用原始图像副本）
            marked_pixmap = current_pixmap.copy()

            # 使用try-finally确保QPainter正确释放
            painter = QPainter(marked_pixmap)
            try:
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(Qt.red, 3))
                painter.drawEllipse(QPoint(int(event.pos().x()), int(event.pos().y())), 5, 5)
            finally:
                painter.end()

            # 更新显示（原子操作）
            self.result_label.setPixmap(marked_pixmap)

            # 保存原始图像引用（防止被垃圾回收）
            self._last_valid_pixmap = current_pixmap

        except Exception as e:
            print(f"显示深度信息时出错: {str(e)}")
            # 出错时恢复原始图像
            if hasattr(self, '_last_valid_pixmap'):
                self.result_label.setPixmap(self._last_valid_pixmap)

    def update_display_mode(self, mode):
        """更新显示模式"""
        self.current_mode = mode
        if mode == "点云":
            self.result_layout.setCurrentIndex(1)  # 切换到点云视图
        else:
            self.result_layout.setCurrentIndex(0)  # 切换到常规结果视图

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if self.capture is not None:
            self.capture.release()
        self.timer.stop()
        event.accept()