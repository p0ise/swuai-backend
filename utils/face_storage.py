import uuid

import numpy as np


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class FaceStorage(metaclass=SingletonMeta):
    def __init__(self):
        self.known_faces = []  # 存储已知人脸的信息，每个条目是一个字典

    def add_known_face(self, face_encoding, name):
        """将新的人脸编码和名称添加到存储中"""
        face_id = str(uuid.uuid4())  # 生成唯一标识符
        face = {
            "id": face_id,
            "name": name,
            "encoding": face_encoding
        }
        self.known_faces.append(face)
        return face

    def match_face(self, face_encoding, tolerance=0.6):
        """在已知的人脸编码中寻找匹配项"""
        if not self.known_faces:
            return None
        distances = np.linalg.norm([face["encoding"] for face in self.known_faces] - face_encoding, axis=1)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < tolerance:
            return self.known_faces[best_match_index]
        return None

    def update_face_encoding(self, face_id, new_encoding):
        """根据人脸ID更新人脸编码"""
        for face in self.known_faces:
            if face["id"] == face_id:
                face["encoding"] = new_encoding
                return True
        return False

    def rename_face(self, face_id, new_name):
        """根据人脸ID更新人脸名称"""
        for face in self.known_faces:
            if face["id"] == face_id:
                face["name"] = new_name
                return True
        return False

    def get_face_info(self, face_id):
        """根据人脸ID获取人脸信息"""
        for face in self.known_faces:
            if face["id"] == face_id:
                return face
        return None
