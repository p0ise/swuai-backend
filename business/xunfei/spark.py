import base64
import hashlib
import hmac
import json
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time

import websocket

from .config import APP_ID, API_KEY_SPARK, API_SECRET_SPARK


class SparkAPI:
    def __init__(self, app_id: str = APP_ID, api_key: str = API_KEY_SPARK, api_secret: str = API_SECRET_SPARK,
                 service_url: str = "wss://spark-api.xf-yun.com/v3.5/chat"):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.host = urlparse(service_url).netloc
        self.path = urlparse(service_url).path
        self.service_url = service_url
        self.domain = self._get_model_domain(self.path)

    def _get_model_domain(self, path: str) -> str:
        domain_map = {
            "/v1.1/chat": "general",
            "/v2.1/chat": "generalv2",
            "/v3.1/chat": "generalv3",
            "/v3.5/chat": "generalv3.5",
        }
        return domain_map.get(path, "unknown")

    def generate_auth_url(self) -> str:
        timestamp = datetime.now()
        formatted_date = format_date_time(mktime(timestamp.timetuple()))
        signature = self._create_signature(formatted_date)
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        url_params = {"authorization": authorization, "date": formatted_date, "host": self.host}
        return f"{self.service_url}?{urlencode(url_params)}"

    def _create_signature(self, date: str) -> str:
        signature_origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 hashlib.sha256).digest()
        return base64.b64encode(signature_sha).decode('utf-8')

    def connect_and_query(self, on_message, messages: list):
        url = self.generate_auth_url()
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(url,
                                    on_message=lambda ws, msg: self.on_message(on_message, msg),
                                    on_error=self.on_error,
                                    on_close=self.on_close,
                                    on_open=lambda ws: self.on_open(ws, messages))
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def on_message(self, on_message, message: str):
        data = json.loads(message)
        on_message(data)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed")

    def on_open(self, ws, messages):
        data = {
            "header": {
                "app_id": self.app_id,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": self.domain,
                    "temperature": 0.5,
                    "max_tokens": 4096,
                    "top_k": 5,
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                    "text": messages
                }
            }
        }
        ws.send(json.dumps(data))
