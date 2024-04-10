import numpy as np

from utils.face_detection import find_faces, find_primary_face, align_face
from utils.face_encoding import encode_face
from utils.face_quality import evaluate_face_quality

# 已知人脸的表示和标签
# TODO: 考虑使用持久化存储替代
known_face_encodings = []  # 存储已知人脸的编码
known_face_names = []  # 存储已知人脸的名称

last_face_boxes = []
last_face_indexes = []


def add_known_face(face_encoding, name):
    global known_face_encodings, known_face_names
    # print("Adding known face: ", name)
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)


def match_face(face_encoding, add_unknown=True):
    global known_face_encodings, known_face_names
    # 比较新的人脸编码与已知人脸编码
    if known_face_encodings:
        # TODO: 如何应对人脸特别多的情况的计算
        # 确保known_face_encodings是NumPy数组
        known_face_encodings_np = np.array(known_face_encodings)
        dists = np.linalg.norm(known_face_encodings_np - face_encoding, axis=1)
        best_match_index = np.argmin(dists)
        if dists[best_match_index] < 0.6:  # 选择合适的阈值
            face_index = int(best_match_index)
            return face_index

    if add_unknown:
        # 为未识别的人脸分配一个唯一名称
        face_index = len(known_face_names)
        name = "未知人脸"
        add_known_face(face_encoding, name)

        return face_index
    else:
        return None


def rename_face(index, name):
    global known_face_names
    if 0 <= index < len(known_face_names):
        known_face_names[index] = name
        return True
    else:
        return False


def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance


# TODO: 考虑有人从某方向离开视野时恰好有人进入的情况
def has_new_face(face_boxes, threshold=50):
    global last_face_boxes
    if len(face_boxes) == 0:
        return False
    # 如果当前帧和上一帧检测到的人脸数量不同，则认为有新的人脸出现
    if len(face_boxes) != len(last_face_boxes):
        return True
    else:
        # 计算当前帧与上一帧中人脸位置的最小距离
        for box in face_boxes:
            is_far = True
            for last_box in last_face_boxes:
                distance = calculate_distance(box, last_box)
                # 如果当前人脸与上一帧中所有人脸的最小距离超过阈值，则认为是新的人脸
                if distance <= threshold:
                    is_far = False
                    break
            if is_far:
                return True

        return False


def recognize_faces(image):
    global last_face_boxes, last_face_indexes, known_face_names
    boxes, probs, landmarks = find_faces(image)

    # TODO: 如果不是按编码顺序而是按位置顺序识别人脸，画面人脸位置交换可能带来与编码不匹配的bug
    if has_new_face(boxes):
        face_indexes = []
        for box in boxes:
            face = image.crop(box)
            face_encoding = encode_face(face)
            face_index = match_face(face_encoding)
            face_indexes.append(face_index)

        last_face_indexes = face_indexes
    else:
        face_indexes = last_face_indexes

    last_face_boxes = boxes

    recognized_faces = []
    for face_index, (left, top, right, bottom), prob, landmark in zip(face_indexes, boxes, probs, landmarks):
        location = (top, right, bottom, left)
        name = known_face_names[face_index]
        recognized_faces.append({
            'index': face_index,
            'location': location,
            'name': name,
            'prob': prob,
            'landmark': landmark.tolist()
        })
    return recognized_faces


def register_face(image, username):
    primary_face = find_primary_face(image)
    if primary_face is not None:
        box, prob, landmark = primary_face
        is_quality_sufficient = evaluate_face_quality(primary_face, prob, image, landmark)
        if is_quality_sufficient:
            # 进行人脸对齐、编码等后续处理
            aligned_face = align_face(image, box, landmark)
            face_encoding = encode_face(aligned_face)
            add_known_face(face_encoding, username)
            return True, "注册成功"
        else:
            return False, "人脸质量不符合要求"
    else:
        return False, "未检测到人脸"


def authenticate_face(image):
    global known_face_names
    primary_face = find_primary_face(image)
    if primary_face is not None:
        box, prob, landmark = primary_face
        is_quality_sufficient = evaluate_face_quality(primary_face, prob, image, landmark)
        if is_quality_sufficient:
            aligned_face = align_face(image, box, landmark)
            face_encoding = encode_face(aligned_face)
            face_index = match_face(face_encoding, add_unknown=False)
            if face_index is None:
                return False, "认证失败，未知人脸"
            else:
                return True, known_face_names[face_index]
        else:
            return False, "人脸质量不符合要求"
    else:
        return False, "未检测到人脸"
