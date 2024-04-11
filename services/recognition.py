import numpy as np

from utils.face_detection import find_faces, align_face
from utils.face_encoding import encode_face
from utils.face_quality import is_face_forward, evaluate_face_quality
from utils.face_storage import FaceStorage

face_storage = FaceStorage()
last_faces = []
last_face_boxes = []


def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance


def has_new_face(face_boxes, threshold=50):
    global last_face_boxes, last_faces
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
    global last_face_boxes, last_faces
    boxes, probs, landmarks = find_faces(image)

    faces = []
    if has_new_face(boxes) or any(face['name'] == "忽略" for face in last_faces):
        for box, prob, landmark in zip(boxes, probs, landmarks):
            is_forward = is_face_forward(landmark)
            face_quality_ok = evaluate_face_quality(box, prob, image, landmark)

            if is_forward and face_quality_ok:
                aligned_face = align_face(image, box, landmark)
                face_encoding = encode_face(aligned_face)
                face = face_storage.match_face(face_encoding)
                if face is None:
                    face = face_storage.add_known_face(face_encoding, "未知")
            else:
                face = {'id': None, 'name': "忽略"}

            faces.append(face)
    else:
        faces = last_faces

    last_face_boxes = boxes
    last_faces = faces

    # 构建识别结果，包括未识别和被忽略的人脸
    recognized_faces = []
    for face, (left, top, right, bottom), prob, landmark in zip(faces, boxes, probs, landmarks):
        face_id = face['id']
        name = face['name']
        location = (top, right, bottom, left)
        recognized_faces.append({
            'id': face_id,
            'location': location,
            'name': name,
            'prob': prob,
            'landmark': landmark.tolist()
        })

    return recognized_faces


def rename_face(face_id, name):
    face_storage.rename_face(face_id, name)
