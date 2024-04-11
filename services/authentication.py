from utils.face_detection import find_primary_face, align_face
from utils.face_encoding import encode_face
from utils.face_quality import evaluate_face_quality
from utils.face_storage import FaceStorage

face_storage = FaceStorage()


def authenticate_face(image):
    """
    对给定的图像进行人脸认证。

    参数:
    - image: 包含用户人脸的图像（PIL.Image格式）。

    返回:
    - 认证成功与否的布尔值以及相应的消息。
    """
    primary_face = find_primary_face(image)
    if primary_face:
        box, prob, landmark = primary_face
        if evaluate_face_quality(box, prob, image, landmark):
            aligned_face = align_face(image, box, landmark)
            face_encoding = encode_face(aligned_face)
            matched_face = face_storage.match_face(face_encoding)
            if matched_face:
                return True, f"认证成功，欢迎 {matched_face['name']}"
            else:
                return False, "认证失败，未知人脸"
        else:
            return False, "人脸质量不符合要求"
    else:
        return False, "未检测到人脸"
