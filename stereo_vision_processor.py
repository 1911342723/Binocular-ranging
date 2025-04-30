import cv2
import numpy as np
from camera_calibrator import CameraCalibrator
from Utils.vision_utils import VisionUtils

"""功能处理"""
class StereoVisionProcessor:
    def __init__(self):
        self.calibrator = CameraCalibrator()
        self.utils = VisionUtils()
        self.stereo = None

    def calibrate_cameras(self, left_image_dir, right_image_dir, chessboard_size=(9, 6), square_size=25.0):
        """执行完整的相机标定流程"""
        objp = self.utils.prepare_chessboard_points(chessboard_size, square_size)
        objpoints = []
        left_imgpoints = []
        right_imgpoints = []

        left_images = self.utils.get_image_paths(left_image_dir)
        right_images = self.utils.get_image_paths(right_image_dir)

        min_images = min(len(left_images), len(right_images))
        if min_images < 4:
            raise RuntimeError(f"需要至少4对图像，当前找到: 左{len(left_images)}张, 右{len(right_images)}张")

        for left_path, right_path in zip(left_images[:min_images], right_images[:min_images]):
            left_img = self.utils.read_image_safe(left_path)
            right_img = self.utils.read_image_safe(right_path)

            if left_img is None or right_img is None:
                continue

            # 查找棋盘格角点
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

            print(f"左图像 {left_path}: 角点检测结果 {ret_left}")
            print(f"右图像 {right_path}: 角点检测结果 {ret_right}")

            if ret_left and ret_right:
                objpoints.append(objp)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                left_imgpoints.append(cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria))
                right_imgpoints.append(cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria))

        print(f"找到的有效图像对数: {len(objpoints)}")

        return self.calibrator.calibrate(objpoints, left_imgpoints, right_imgpoints)

    # 在 stereo_vision_processor.py 中检查是否正确初始化了 stereo 匹配器
    def init_stereo_matcher(self):
        if not hasattr(self, 'stereo') or self.stereo is None:
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=1,
                numDisparities=64,
                blockSize=3,
                P1=8 * 3 * 3 * 3,
                P2=32 * 3 * 3 * 3,
                disp12MaxDiff=-1,
                preFilterCap=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=100,
                mode=cv2.STEREO_SGBM_MODE_HH)

    def process_frame(self, frame):
        """处理视频帧"""
        if not self.calibrator.is_calibrated:
            raise RuntimeError("请先完成相机标定！")

        if self.stereo is None:
            self.init_stereo_matcher()

        # 验证输入帧
        if frame is None or frame.size == 0:
            raise ValueError("输入帧无效")
        if frame.shape[0] != 480 or frame.shape[1] != 1280:
            frame = cv2.resize(frame, (1280, 480))

        try:
            # 分割左右图像
            frame1 = frame[0:480, 0:640]  # 左图
            frame2 = frame[0:480, 640:1280]  # 右图

            # 转换为灰度图并校正
            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # 从校准器中获取参数
            size = self.calibrator.size
            left_map = cv2.initUndistortRectifyMap(
                self.calibrator.left_camera_matrix,  # 修改这里
                self.calibrator.left_distortion,  # 修改这里
                self.calibrator.R1,  # 修改这里
                self.calibrator.P1,  # 修改这里
                size,
                cv2.CV_16SC2
            )
            img1_rectified = cv2.remap(imgL, left_map[0], left_map[1], cv2.INTER_LINEAR)

            right_map = cv2.initUndistortRectifyMap(
                self.calibrator.right_camera_matrix,  # 修改这里
                self.calibrator.right_distortion,  # 修改这里
                self.calibrator.R2,  # 修改这里
                self.calibrator.P2,  # 修改这里
                size,
                cv2.CV_16SC2
            )
            # 打印标定参数检查合理性
            print("Left Camera Matrix:\n", self.calibrator.left_camera_matrix)
            print("Right Camera Matrix:\n", self.calibrator.right_camera_matrix)
            print("Translation Vector T:\n", self.calibrator.T)


            img2_rectified = cv2.remap(imgR, right_map[0], right_map[1], cv2.INTER_LINEAR)

            # 计算视差
            disparity = self.stereo.compute(img1_rectified, img2_rectified)

            # 计算3D坐标（使用标定器的Q矩阵）
            threeD = cv2.reprojectImageTo3D(disparity, self.calibrator.Q, handleMissingValues=True)  # 修改这里
            threeD = threeD * 16  # 缩放因子

            # 生成灰度图和深度图
            gray_img = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
            depth_img = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            return frame1, gray_img, depth_img, threeD
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            raise