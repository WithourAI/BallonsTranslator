import cv2
import requests
import json
import base64
import numpy as np


class StariverOCR:
    def __init__(self, token):
        self.token = token
        self.url = 'https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/manga_ocr'

    def ocr(self, img: np.ndarray):
        img = cv2.imencode('.png', img)[1]
        img_base64 = base64.b64encode(img).decode('utf-8')
        data = {
            "token": self.token,
            "mask": True,
            "refine": True,
            "filtrate": True,
            "disable_skip_area": True,
            "detect_scale": 3,
            "merge_threshold": 0.5,
            "low_accuracy_mode": False,
            "image": img_base64
        }
        response = requests.post(self.url, data=json.dumps(data))
        text_blocks = response.json()['Data']['text_block']
        texts = [text for block in text_blocks for text in block['texts']]
        return texts
