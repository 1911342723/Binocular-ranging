import cv2
import numpy as np
from Utils.vision_utils import VisionUtils


class CameraCalibrator:
    def __init__(self, default_left_camera_matrix=None, default_right_camera_matrix=None,
                 default_left_distortion=None, default_right_distortion=None,
                 default_R=None, default_T=None, image_size=(1280, 480)):
        """
        初始化相机标定器，并设置默认的相机参数。
        """
        self.left_camera_matrix = np.array(default_left_camera_matrix,
                                           dtype=np.float32) if default_left_camera_matrix is not None else None
        self.right_camera_matrix = np.array(default_right_camera_matrix,
                                            dtype=np.float32) if default_right_camera_matrix is not None else None
        self.left_distortion = np.array(default_left_distortion,
                                        dtype=np.float32) if default_left_distortion is not None else None
        self.right_distortion = np.array(default_right_distortion,
                                         dtype=np.float32) if default_right_distortion is not None else None
        self.R = np.array(default_R, dtype=np.float32) if default_R is not None else None
        self.T = np.array(default_T, dtype=np.float32) if default_T is not None else None
        self.size = image_size
        self.is_calibrated = True

        # 立体校正参数
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.E = None
        self.F = None

        # 工具类
        self.utils = VisionUtils()

    def set_default_parameters(self, left_camera_matrix, right_camera_matrix,
                               left_distortion, right_distortion, R, T):
        """设置默认的相机参数"""
        self.left_camera_matrix = np.array(left_camera_matrix, dtype=np.float32)
        self.right_camera_matrix = np.array(right_camera_matrix, dtype=np.float32)
        self.left_distortion = np.array(left_distortion, dtype=np.float32)
        self.right_distortion = np.array(right_distortion, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.T = np.array(T, dtype=np.float32)

        self.is_calibrated = True

    def calibrate(self, objpoints, left_imgpoints, right_imgpoints, use_initial_guess=True):
        """执行双目相机标定"""
        if self.left_camera_matrix is None or self.right_camera_matrix is None:
            raise ValueError("左或右相机内参矩阵未设置。请先设置默认参数或通过标定获取。")

        # 畸变系数应该是1x5的数组
        if (self.left_distortion is None or len(self.left_distortion) != 5 or
                self.right_distortion is None or len(self.right_distortion) != 5):
            raise ValueError("畸变系数应该是1x5的数组。")

        # 标志位：是否使用初始猜测
        flags = 0
        if use_initial_guess:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        # 执行双目标定
        ret, self.left_camera_matrix, self.left_distortion, \
            self.right_camera_matrix, self.right_distortion, \
            self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints, left_imgpoints, right_imgpoints,
            self.left_camera_matrix,
            self.left_distortion,
            self.right_camera_matrix,
            self.right_distortion,
            self.size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            flags=flags)

        # 立体校正
        if ret < 1.0:  # 标定成功
            self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
                self.left_camera_matrix, self.left_distortion,
                self.right_camera_matrix, self.right_distortion,
                self.size, self.R, self.T)

            self.is_calibrated = True
        else:
            self.is_calibrated = False

        return ret

    def set_camera_parameters(self, left_camera_matrix, right_camera_matrix,
                              left_distortion, right_distortion, R, T):
        """直接设置相机参数（用于模板输入）"""
        self.left_camera_matrix = np.array(left_camera_matrix, dtype=np.float32)
        self.right_camera_matrix = np.array(right_camera_matrix, dtype=np.float32)
        self.left_distortion = np.array(left_distortion, dtype=np.float32)
        self.right_distortion = np.array(right_distortion, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.T = np.array(T, dtype=np.float32)

        # 计算其他必要矩阵
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            self.size, self.R, self.T)

        self.is_calibrated = True
        return True

    def calibrate_with_parameters(self):
        """使用已设置的参数进行标定验证（可选）"""
        if not all([self.left_camera_matrix, self.right_camera_matrix,
                    self.left_distortion, self.right_distortion,
                    self.R, self.T]):
            return False

        # 计算立体校正参数
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            self.size, self.R, self.T)

        self.is_calibrated = True
        return True