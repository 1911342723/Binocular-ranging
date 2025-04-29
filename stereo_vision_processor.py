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

            # 转换为灰度图
            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # 预计算校正映射（优化性能）
            if not hasattr(self, 'map1x'):
                self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                    self.calibrator.left_camera_matrix,
                    self.calibrator.left_distortion,
                    self.calibrator.R1,
                    self.calibrator.P1,
                    self.calibrator.size,
                    cv2.CV_32FC1)  # 改用32位浮点提高精度

                self.map2x, self.map2y = cv2.initUndistortRectifyMap(
                    self.calibrator.right_camera_matrix,
                    self.calibrator.right_distortion,
                    self.calibrator.R2,
                    self.calibrator.P2,
                    self.calibrator.size,
                    cv2.CV_32FC1)

            # 执行图像校正（修正参数顺序）
            img1_rectified = cv2.remap(
                imgL,
                self.map1x,
                self.map1y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)

            img2_rectified = cv2.remap(
                imgR,
                self.map2x,
                self.map2y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)


            # 计算视差
            disparity = self.stereo.compute(img1_rectified, img2_rectified)

            # 计算3D坐标
            threeD = cv2.reprojectImageTo3D(
                disparity,
                self.calibrator.Q,
                handleMissingValues=True)
            threeD = threeD * 16  # 缩放因子

            # 生成可视化结果
            gray_img = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
            depth_img = cv2.normalize(
                disparity, None, 0, 255,
                cv2.NORM_MINMAX, cv2.CV_8U)
            depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

            return frame1, gray_img, depth_img, threeD

        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            raise