import cv2
import numpy as np
import os
import glob
import re
"""工具类"""
class VisionUtils:
    @staticmethod
    def get_image_paths(directory):
        """获取目录中所有支持的图像文件（兼容中文路径和多种命名格式）"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        directory = os.path.normpath(directory)

        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return []

        for ext in extensions:
            try:
                files.extend(glob.glob(os.path.join(directory, ext), recursive=True))
            except Exception as e:
                print(f"搜索图像时出错: {str(e)}")
                continue

        def extract_number(filename):
            base = os.path.basename(filename)
            numbers = re.findall(r'\d+', base)
            return int(numbers[-1]) if numbers else 0

        return sorted(files, key=extract_number)

    @staticmethod
    def read_image_safe(path):
        """安全读取图像（解决中文路径问题）"""
        try:
            with open(path, 'rb') as f:
                img_data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                return img if img is not None else None
        except Exception as e:
            print(f"读取图像错误 {path}: {str(e)}")
            return None

    @staticmethod
    def prepare_chessboard_points(chessboard_size=(9, 6), square_size=25.0):
        """准备棋盘格角点"""
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
        return objp

class CameraParamsParser:
    @staticmethod
    def parse_matrix(matrix_str):
        """
        将字符串形式的矩阵转换为 numpy 数组
        :param matrix_str: 矩阵的字符串表示，如 "[[1, 2], [3, 4]]"
        :return: numpy 数组
        """
        try:
            # 去除多余的空格和换行符
            matrix_str = matrix_str.replace('\n', '').replace(' ', '')
            matrix = eval(matrix_str)
            return np.array(matrix)
        except Exception as e:
            print(f"解析矩阵时出错: {e}")
            return None

    @staticmethod
    def parse_vector(vector_str):
        """
        将字符串形式的向量转换为 numpy 数组
        :param vector_str: 向量的字符串表示，如 "[1, 2, 3]"
        :return: numpy 数组
        """
        try:
            # 去除多余的空格和换行符
            vector_str = vector_str.replace('\n', '').replace(' ', '')
            vector = eval(vector_str)
            return np.array(vector)
        except Exception as e:
            print(f"解析向量时出错: {e}")
            return None

    @staticmethod
    def parse_params(params_str):
        """
        解析包含多个相机参数的字符串
        :param params_str: 包含多个相机参数的字符串，每个参数用换行分隔，格式为 "参数名 = 矩阵/向量字符串"
        :return: 包含解析后参数的字典
        """
        params = {}
        lines = params_str.strip().split('\n')
        for line in lines:
            if '=' in line:
                param_name, param_value_str = line.split('=', 1)
                param_name = param_name.strip()
                param_value_str = param_value_str.strip()
                if '[' in param_value_str and ']' in param_value_str:
                    if param_value_str.count('[') > 1:
                        # 矩阵
                        param_value = CameraParamsParser.parse_matrix(param_value_str)
                    else:
                        # 向量
                        param_value = CameraParamsParser.parse_vector(param_value_str)
                    if param_value is not None:
                        params[param_name] = param_value
        return params