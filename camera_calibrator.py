import cv2
import numpy as np
import os

"""相机标定"""
class CameraCalibrator:
    def __init__(self):
        self.left_camera_matrix = None
        self.right_camera_matrix = None
        self.left_distortion = None
        self.right_distortion = None
        self.R = None
        self.T = None
        self.size = (640, 480)
        self.is_calibrated = False

    def calibrate(self, objpoints, left_imgpoints, right_imgpoints):
        """执行双目相机标定"""
        # 单目标定
        ret, self.left_camera_matrix, self.left_distortion, _, _ = cv2.calibrateCamera(
            objpoints, left_imgpoints, self.size, None, None)

        ret, self.right_camera_matrix, self.right_distortion, _, _ = cv2.calibrateCamera(
            objpoints, right_imgpoints, self.size, None, None)

        # 双目标定
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, self.R, self.T, _, _ = cv2.stereoCalibrate(
            objpoints, left_imgpoints, right_imgpoints,
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            self.size, flags=flags)

        # 立体校正
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            self.size, self.R, self.T)

        self.is_calibrated = True
        return ret