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

from Utils.vision_utils import VisionUtils
from stereo_vision_processor import StereoVisionProcessor
from calibration_dialog import CalibrationDialog
from Utils.template_input_utils import TemplateInputDialog

"""整体窗口的布局"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("双目视觉测距系统")
        self.setGeometry(100, 100, 1200, 800)

        # 默认相机参数
        self.default_left_camera_matrix = [
            [516.5066236, -1.444673028, 320.2950423],
            [0, 516.5816117, 270.7881873],
            [0., 0., 1.]
        ]
        self.default_right_camera_matrix = [
            [511.8428182, 1.295112628, 317.310253],
            [0, 513.0748795, 269.5885026],
            [0., 0., 1.]
        ]
        self.default_left_distortion = [
            -0.046645194, 0.077595167, 0.012476819, -0.000711358, 0
        ]  # 注意：畸变系数应该是1x5的列表
        self.default_right_distortion = [
            -0.061588946, 0.122384376, 0.011081232, -0.000750439, 0
        ]  # 注意：畸变系数应该是1x5的列表
        self.default_R = [
            [0.999911333, -0.004351508, 0.012585312],
            [0.004184066, 0.999902792, 0.013300386],
            [-0.012641965, -0.013246549, 0.999832341]
        ]
        self.default_T = [-120.3559901, -0.188953775, -0.662073075]

        # 转换为正确的numpy数组格式
        self.default_R = np.array(self.default_R, dtype=np.float32)
        self.default_T = np.array(self.default_T, dtype=np.float32).reshape(3, 1)  # T应为列向量

        # 初始化处理器，并设置默认参数
        self.processor = StereoVisionProcessor()
        # 初始化处理器时传递正确格式
        self.processor.calibrator.set_default_parameters(
            self.default_left_camera_matrix,
            self.default_right_camera_matrix,
            self.default_left_distortion,
            self.default_right_distortion,
            self.default_R,  # 现在为np.float32矩阵
            self.default_T  # 现在为np.float32列向量
        )
        self.processor.calibrator.size = (1280, 480)  # 根据实际情况设置图像尺寸

        # 视频捕获改为空初始化
        self.capture = None  # 原初始化代码中的capture初始化删除

        self.timer = QTimer()  # 先初始化定时器
        self.current_mode = "深度图"
        self.setup_ui()  # 最后初始化UI




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
        self.calib_status = QLabel("状态: 已标定（使用默认参数）")

        btn_layout = QHBoxLayout()

        # 标定相机按钮（如果需要重新标定）
        self.calibrate_btn = QPushButton("重新标定相机")
        self.calibrate_btn.clicked.connect(self.show_calibration_dialog)

        # 相机参数设置按钮
        self.param_btn = QPushButton("相机参数设置")
        self.param_btn.clicked.connect(self.show_template_input)

        # 修改按钮名称和连接
        self.upload_btn = QPushButton("上传视频")  # 原start_btn改为upload_btn
        self.upload_btn.clicked.connect(self.upload_video)  # 连接新的事件处理函数

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_video)

        # 修改按钮布局
        btn_layout.addWidget(self.calibrate_btn)
        btn_layout.addWidget(self.param_btn)
        btn_layout.addWidget(self.upload_btn)  # 替换原来的start_btn
        btn_layout.addWidget(self.stop_btn)



        right_layout.addWidget(self.calib_status)
        right_layout.addLayout(btn_layout)

        # 合并布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def show_calibration_dialog(self):
        """显示标定对话框"""
        dialog = CalibrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 假设 CalibrationDialog 在接受时会返回必要的参数
            # 这里需要根据实际情况获取参数
            # 例如：
            # left_dir = dialog.get_left_directory()
            # right_dir = dialog.get_right_directory()
            # chessboard_size = dialog.get_chessboard_size()
            # square_size = dialog.get_square_size()
            # self.start_calibration(left_dir, right_dir, chessboard_size, square_size)
            pass  # 你需要根据 CalibrationDialog 的实现来获取参数

    def start_calibration(self, left_dir, right_dir, chessboard_size, square_size):
        """执行相机标定"""
        try:
            # 读取棋盘格角点
            objp = VisionUtils.prepare_chessboard_points(chessboard_size, square_size)
            objpoints = []  # 3D点集
            left_imgpoints = []  # 左相机图像点集
            right_imgpoints = []  # 右相机图像点集

            left_images = VisionUtils.get_image_paths(left_dir)
            right_images = VisionUtils.get_image_paths(right_dir)

            min_images = min(len(left_images), len(right_images))
            if min_images < 4:
                raise RuntimeError(f"需要至少4对图像，当前找到: 左{len(left_images)}张, 右{len(right_images)}张")

            for left_path, right_path in zip(left_images[:min_images], right_images[:min_images]):
                left_img = VisionUtils.read_image_safe(left_path)
                right_img = VisionUtils.read_image_safe(right_path)

                if left_img is None or right_img is None:
                    continue

                gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

                ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
                ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

                if ret_left and ret_right:
                    objpoints.append(objp)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    left_imgpoints.append(cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria))
                    right_imgpoints.append(cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria))

            if not objpoints:
                raise RuntimeError("未找到有效的棋盘格角点")

            print(f"找到的有效图像对数: {len(objpoints)}")

            # 执行标定
            ret = self.processor.calibrator.calibrate(
                objpoints, left_imgpoints, right_imgpoints, use_initial_guess=True
            )



            if ret is not None:  # 假设 ret 是标定误差
                self.calib_status.setText(f"状态: 已标定 (误差: {ret:.2f})")
                QMessageBox.information(self, "标定成功", f"标定完成，RMS误差: {ret:.2f}")
            else:
                self.calib_status.setText("状态: 标定失败")
                QMessageBox.warning(self, "标定失败", "请检查图像和参数设置")
        except Exception as e:
            self.calib_status.setText("状态: 标定错误")
            QMessageBox.critical(self, "标定错误", str(e))

    def show_template_input(self):
        """显示模板参数输入对话框"""
        dialog = TemplateInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 获取输入的参数
            left_camera_matrix_text = dialog.left_camera_matrix.text()
            right_camera_matrix_text = dialog.right_camera_matrix.text()
            left_distortion_text = dialog.left_distortion.text()
            right_distortion_text = dialog.right_distortion.text()
            R_text = dialog.R.text()
            T_text = dialog.T.text()

            try:
                # 解析输入的参数
                left_camera_matrix = self.parse_matrix(left_camera_matrix_text, (3, 3))
                right_camera_matrix = self.parse_matrix(right_camera_matrix_text, (3, 3))
                left_distortion = self.parse_vector(left_distortion_text, 5)
                right_distortion = self.parse_vector(right_distortion_text, 5)
                R = self.parse_matrix(R_text, (3, 3))
                T = self.parse_vector(T_text, 3)

                # 设置相机参数
                self.processor.calibrator.set_camera_parameters(
                    left_camera_matrix.tolist(), right_camera_matrix.tolist(),
                    left_distortion.tolist(), right_distortion.tolist(),
                    R.tolist(), T.tolist()
                )

                self.calib_status.setText("状态: 已标定（使用新参数）")
                QMessageBox.information(self, "成功", "相机参数已成功设置！")

            except ValueError as e:
                QMessageBox.critical(self, "错误", f"参数解析失败: {str(e)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"发生错误: {str(e)}")

    def parse_matrix(self, text, shape):
        try:
            # 移除可能的方括号和换行符，然后按逗号分割
            values = [float(x.strip()) for x in text.replace('[', '').replace(']', '').replace('\n', '').split(',') if
                      x.strip()]
            if len(values) != shape[0] * shape[1]:
                raise ValueError(f"需要 {shape[0] * shape[1]} 个值")
            return np.array(values).reshape(shape)
        except Exception as e:
            raise ValueError(f"解析矩阵失败: {str(e)}")

    def parse_vector(self, text, size):
        try:
            values = [float(x.strip()) for x in text.replace('[', '').replace(']', '').replace('\n', '').split(',') if
                      x.strip()]
            if len(values) != size:
                raise ValueError(f"需要 {size} 个值")
            return np.array(values).reshape(1, -1)
        except Exception as e:
            raise ValueError(f"解析向量失败: {str(e)}")

    def upload_video(self):
        """上传视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "Video Files (*.avi *.mp4 *.mov *.mkv);;All Files (*)"
        )

        if not file_path:
            return

        # 释放原有视频资源
        if self.capture and self.capture.isOpened():
            self.capture.release()

        # 初始化新视频
        self.capture = cv2.VideoCapture(file_path)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件！")
            return

        # 自动开始播放
        self.start_processing()

    def start_processing(self):
        """开始视频处理（原start_video改名）"""
        # if not self.processor.calibrator.is_calibrated:
        #     QMessageBox.warning(self, "警告", "请先完成相机标定！")
        #     return

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms更新一帧

    def stop_video(self):
        """停止视频处理"""
        self.timer.stop()
        if self.capture and self.capture.isOpened():
            self.capture.release()

    def update_frame(self):
        """更新视频帧"""
        try:
            ret, frame = self.capture.read()
            if not ret:
                self.timer.stop()
                QMessageBox.warning(self, "警告", "视频结束或无法读取帧！")
                return

            # 确保帧尺寸符合预期，如果需要可以调整
            if frame.shape[1] != 1280 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (1280, 480))

            # 处理帧
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
            self.timer.stop()
            QMessageBox.critical(self, "错误", f"处理帧时出错: {str(e)}")

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

    def sync_calibration_status(self):
        """同步标定状态到界面显示"""
        if self.processor.calibrator.is_calibrated:
            self.calib_status.setText("状态: 已标定（参数已加载）")
            self.calib_status.setStyleSheet("color: green;")
        else:
            self.calib_status.setText("状态: 未标定")
            self.calib_status.setStyleSheet("color: red;")
