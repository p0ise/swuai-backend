import base64
import datetime
import json
from urllib.parse import urlencode

import requests

from .config import APP_ID, API_KEY_FACE_COMPARE, API_SECRET_FACE_COMPARE, SERVER_ID_FACE_COMPARE
from .utils import parse_url, generate_signature, assemble_authorization_header, encode_image


class FaceCompareClient:
    def __init__(self):
        self.base_url = f'http://api.xf-yun.com/v1/private/{SERVER_ID_FACE_COMPARE}'

    def generate_request_url(self):
        schema, host, path = parse_url(self.base_url)
        date = format(datetime.utcnow(), '%a, %d %b %Y %H:%M:%S GMT')
        signature_sha = generate_signature(API_SECRET_FACE_COMPARE, host, date, "POST", path)
        authorization = assemble_authorization_header(API_KEY_FACE_COMPARE, signature_sha)
        values = {"host": host, "date": date, "authorization": authorization}
        return f"{self.base_url}?{urlencode(values)}"

    def compare_faces(self, img1_path, img2_path):
        request_url = self.generate_request_url()
        headers = {'content-type': "application/json", 'host': 'api.xf-yun.com', 'app_id': APP_ID}

        img1_encoded, img1_type = encode_image(img1_path)
        img2_encoded, img2_type = encode_image(img2_path)

        body = json.dumps({
            "header": {"app_id": APP_ID, "status": 3},
            "parameter": {SERVER_ID_FACE_COMPARE: {"service_kind": "face_compare",
                                                   "face_compare_result": {"encoding": "utf8", "compress": "raw",
                                                                           "format": "json"}}},
            "payload": {
                "input1": {"encoding": img1_type, "status": 3, "image": img1_encoded},
                "input2": {"encoding": img2_type, "status": 3, "image": img2_encoded}
            }
        })

        response = requests.post(request_url, data=body, headers=headers)
        if response.status_code == 200:
            result = json.loads(response.text)
            code = result['header']['code']
            if code == 0:
                data = json.loads(base64.b64decode(result['payload']['face_compare_result']['text']))['score']
            else:
                data = result['header']['message']
        else:
            code = -1
            data = '请求错误'

        return code, data
