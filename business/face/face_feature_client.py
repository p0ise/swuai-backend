import base64
import hashlib
import os.path
import time
from urllib.parse import urljoin

import requests

from .config import APP_ID, API_KEY_FACE_FEATURE


class FaceFeatureClient:
    def __init__(self):
        self.base_url = "http://tupapi.xfyun.cn/v1/"
        self.types = ['age', 'sex', 'expression', 'face_score']

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

    def analyze(self, type, image_url):
        url = urljoin(self.base_url, type)
        image_name = image_url.split('/')[-1]
        if os.path.exists(image_url):
            headers = self.get_header(image_name)
            with open(image_name, 'rb') as f:
                data = f.read()
        else:
            data = None
            headers = self.get_header(image_name, image_url)
        response = requests.post(url, data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            code = result['code']
            if result['code'] == 0:
                value = result['data']['file_list'][0]['label']
            else:
                value = result['desc']
        else:
            code = -1
            value = '请求错误'

        return code, value

    def analyze_all(self, image_url):
        results = []
        for type in self.types:
            code, value = self.analyze(type, image_url)
            if code == 0:
                results.append({type: value})
        return results
