import base64
import hashlib
import hmac
import mimetypes


class AssembleHeaderException(Exception):
    pass


def parse_url(request_url):
    stidx = request_url.index("://")
    host = request_url[stidx + 3:]
    schema = request_url[:stidx + 3]
    edidx = host.index("/")
    if edidx <= 0:
        raise AssembleHeaderException("Invalid request URL: " + request_url)
    path = host[edidx:]
    host = host[:edidx]
    return schema, host, path


def generate_signature(api_secret, host, date, method, path):
    signature_origin = f"host: {host}\ndate: {date}\n{method} {path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    return base64.b64encode(signature_sha).decode('utf-8')


def assemble_authorization_header(api_key, signature_sha):
    authorization_origin = (f'api_key="{api_key}", algorithm="hmac-sha256", '
                            f'headers="host date request-line", signature="{signature_sha}"')
    return base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')


def encode_image(image_path):
    """
    对图片文件进行Base64编码，并自动检测图片类型。
    返回编码后的图片字符串和图片类型。
    """
    # 检测文件类型
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Cannot determine the file type of the image.")
    # 获取文件的扩展名，例如 "image/jpeg" -> "jpeg"
    file_type = mime_type.split('/')[-1]

    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string, file_type
