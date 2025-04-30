import cv2
import numpy as np
from camera_calibrator import CameraCalibrator
from Utils.vision_utils import VisionUtils


class StereoVisionProcessor:
    def __init__(self):
        self.calibrator = CameraCalibrator()
        self.utils = VisionUtils()
        self.stereo = None
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None
        self.is_rectified = False

    def calibrate_cameras(self, left_image_dir=None, right_image_dir=None,
                          chessboard_size=(9, 6), square_size=25.0):
        """执行完整的相机标定流程"""
        if left_image_dir is None and right_image_dir is None:
            # 如果没有提供图像目录，则使用已设置的参数
            if not self.calibrator.is_calibrated:
                raise RuntimeError("没有提供图像目录且未设置相机参数")
            return self.calibrator.calibrate_with_parameters()

        if not self.calibrator.is_calibrated:
            # 这里改为调用 self.utils.prepare_chessboard_points()
            objp = self.utils.prepare_chessboard_points(chessboard_size, square_size)
            objpoints = []
            left_imgpoints = []
            right_imgpoints = []

            left_images = self.utils.get_image_paths(left_image_dir) if left_image_dir else []
            right_images = self.utils.get_image_paths(right_image_dir) if right_image_dir else []

            # 确保左右图像数量匹配
            min_images = min(len(left_images), len(right_images))
            if min_images < 4:
                raise RuntimeError(f"需要至少4对图像，当前找到: 左{len(left_images)}张, 右{len(right_images)}张")

            for left_path, right_path in zip(left_images[:min_images], right_images[:min_images]):
                left_img = self.utils.read_image_safe(left_path)
                right_img = self.utils.read_image_safe(right_path)

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
            return self.calibrator.calibrate(objpoints, left_imgpoints, right_imgpoints)
        else:
            return self.calibrator.calibrate_with_parameters()

    def init_stereo_matcher(self):
        """初始化立体匹配器"""
        if self.stereo is None:
            # SGBM参数
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16 * 5,  # 必须是16的倍数
                blockSize=5,
                P1=8 * 3 * 5 ** 2,
                P2=32 * 3 * 5 ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32)

    def process_frame(self, frame):
        """处理视频帧"""
        if self.stereo is None:
            self.init_stereo_matcher()

        if frame is None or frame.size == 0:
            raise ValueError("输入帧无效")
        if frame.shape[0] != self.calibrator.size[1] or frame.shape[1] != self.calibrator.size[0]:
            frame = cv2.resize(frame, self.calibrator.size)

        # 立体校正
        self.size = (640, 480)
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            self.size, self.R, self.T
        )
        self.is_rectified = True
        try:
            # 分割左右图像
            frame_height, frame_width, _ = frame.shape
            if frame_width % 2 != 0:
                frame_width -= 1  # 确保宽度是偶数
            left_frame = frame[:, :frame_width // 2]
            right_frame = frame[:, frame_width // 2:]

            if left_frame.shape[1] != frame_width // 2 or right_frame.shape[1] != frame_width // 2:
                raise ValueError("图像分割后尺寸不正确")

            imgL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # 使用预计算的映射（如果已计算）
            if self.map1x is not None and self.map1y is not None:
                img1_rectified = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
                img2_rectified = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)
            else:
                # 计算并存储映射
                if not self.is_rectified:
                    if (self.calibrator.R1 is None or self.calibrator.R2 is None or
                            self.calibrator.P1 is None or self.calibrator.P2 is None):
                        raise ValueError("立体校正参数未计算")

                    self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                        self.calibrator.left_camera_matrix, self.calibrator.left_distortion,
                        self.calibrator.R1, self.calibrator.P1,
                        self.calibrator.size, cv2.CV_16SC2)
                    self.map2x, self.map2y = cv2.initUndistortRectifyMap(
                        self.calibrator.right_camera_matrix, self.calibrator.right_distortion,
                        self.calibrator.R2, self.calibrator.P2,
                        self.calibrator.size, cv2.CV_16SC2)
                    self.is_rectified = True

                img1_rectified = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
                img2_rectified = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)

            # 计算视差
            disparity = self.stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0

            # 计算3D坐标
            threeD = cv2.reprojectImageTo3D(disparity, self.calibrator.Q)

            # 生成灰度图和深度图
            gray_img = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
            depth_img = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

            return frame_height, frame_width, left_frame, right_frame, img1_rectified, img2_rectified, disparity, threeD, gray_img, depth_img

        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            raise