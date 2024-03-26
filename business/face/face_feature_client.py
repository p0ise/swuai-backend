import base64
import hashlib
import time

import requests

from config import APP_ID, API_KEY_FACE_FEATURE


class FaceFeatureClient:
    def __init__(self):
        self.url_age = "http://tupapi.xfyun.cn/v1/age"
        self.url_face_score = "http://tupapi.xfyun.cn/v1/face_score"
        self.url_sex = "http://tupapi.xfyun.cn/v1/sex"
        self.url_expression = "http://tupapi.xfyun.cn/v1/expression"

    def get_header(self, image_name, image_url=None):
        cur_time = str(int(time.time()))
        param = {"image_name": image_name, "image_url": image_url}
        param_base64 = base64.b64encode(str(param).encode('utf-8'))

        m2 = hashlib.md5()
        m2.update((API_KEY_FACE_FEATURE + cur_time + str(param_base64, 'utf-8')).encode('utf-8'))
        check_sum = m2.hexdigest()

        header = {
            'X-CurTime': cur_time,
            'X-Param': param_base64,
            'X-Appid': APP_ID,
            'X-CheckSum': check_sum,
        }
        return header

    def analyze_age(self, image_name, image_url=None):
        headers = self.get_header(image_name, image_url)
        response = requests.post(self.url_age, headers=headers)
        return response.json()

    def analyze_face_score(self, image_name, image_url=None):
        headers = self.get_header(image_name, image_url)
        response = requests.post(self.url_face_score, headers=headers)
        return response.json()

    def analyze_sex(self, image_name, image_url=None):
        headers = self.get_header(image_name, image_url)
        response = requests.post(self.url_sex, headers=headers)
        return response.json()

    def analyze_expression(self, image_name, image_url=None):
        headers = self.get_header(image_name, image_url)
        response = requests.post(self.url_expression, headers=headers)
        return response.json()
