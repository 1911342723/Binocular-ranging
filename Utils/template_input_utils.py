import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit, QPushButton, QMessageBox
from Utils.vision_utils import CameraParamsParser

class TemplateInputDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("模板化输入")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.input_text_edit = QPlainTextEdit()
        self.input_text_edit.setPlaceholderText("请输入符合格式要求的相机参数...")
        layout.addWidget(self.input_text_edit)
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.process_template_input)
        layout.addWidget(ok_btn)
        self.setLayout(layout)

    def process_template_input(self):
        params_str = self.input_text_edit.toPlainText()
        if not params_str.strip():
            QMessageBox.warning(self, "输入错误", "输入不能为空，请重新输入。")
            return
        try:
            parsed_params = CameraParamsParser.parse_params(params_str)
            self.main_window.apply_template_params(parsed_params)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "解析错误", f"参数解析失败，原因：{str(e)}")