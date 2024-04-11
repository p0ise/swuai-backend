from utils.face_detection import find_primary_face, align_face
from utils.face_encoding import encode_face
from utils.face_quality import is_face_forward, evaluate_face_quality
from utils.face_storage import FaceStorage

face_storage = FaceStorage()


def register_face(image, username):
    """
    注册新的用户人脸。

    参数:
    - image: 包含用户人脸的图像（PIL.Image格式）。
    - username: 用户的名称。

    返回:
    - 成功或失败的消息。
    """
    primary_face = find_primary_face(image)
    if primary_face:
        box, prob, landmark = primary_face
        if is_face_forward(landmark) and evaluate_face_quality(box, prob, image, landmark):
            aligned_face = align_face(image, box, landmark)
            face_encoding = encode_face(aligned_face)
            matched_face = face_storage.match_face(face_encoding)
            if matched_face is None:
                face = face_storage.add_known_face(face_encoding, username)
                face_id = face['id']
                return True, f"用户 {username} 注册成功！人脸ID：{face_id}"
            else:
                return False, "人脸已注册"
        else:
            return False, "人脸质量不符合要求或非正脸"
    else:
        return False, "未检测到人脸"
