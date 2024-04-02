import base64

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化MTCNN和InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 已知人脸的表示和标签
# TODO: 考虑使用持久化存储替代
known_face_encodings = []  # 存储已知人脸的编码
known_face_names = []  # 存储已知人脸的名称

last_face_boxes = []
last_face_indexes = []


# 将图像从OpenCV格式转换为PIL格式
def cv2_to_pil(image):
    # OpenCV图像是BGR格式，转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def parse_frame_data(frame_data):
    frame_data = base64.b64decode(frame_data.split(',')[1])
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame, flags=1)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2_to_pil(small_frame)

    return rgb_small_frame


def find_faces(image):
    # 使用MTCNN检测人脸
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return []
    else:
        return boxes


def encode_face(face):
    # 图像预处理转换
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),  # 调整图像大小以匹配模型期望的输入
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
    ])
    image_tensor = preprocess(face).unsqueeze(0).to(device)
    # 使用Inception Resnet进行人脸编码
    with torch.no_grad():
        face_embedding = resnet(image_tensor)
    # face_embedding = resnet(face.unsqueeze(0).to(device))
    face_encoding = face_embedding.detach().cpu().numpy()[0]
    return face_encoding


def match_face(face_encoding):
    # 比较新的人脸编码与已知人脸编码
    if known_face_encodings:
        # TODO: 如何应对人脸特别多的情况的计算
        # 确保known_face_encodings是NumPy数组
        known_face_encodings_np = np.array(known_face_encodings)
        dists = np.linalg.norm(known_face_encodings_np - face_encoding, axis=1)
        best_match_index = np.argmin(dists)
        if dists[best_match_index] < 0.9:  # 选择合适的阈值
            face_index = int(best_match_index)
            return face_index

    # 为未识别的人脸分配一个唯一名称
    face_index = len(known_face_names)
    name = "未知人脸"  # 使用定长序号生成名称
    add_known_face(face_encoding, name)

    return face_index


def add_known_face(face_encoding, name):
    global known_face_encodings, known_face_names
    # print("Adding known face: ", name)
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)


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
    face_boxes = find_faces(image)

    # TODO: 如果不是按编码顺序而是按位置顺序识别人脸，画面人脸位置交换可能带来与编码不匹配的bug
    if has_new_face(face_boxes):
        # print("New face detected")
        face_indexes = []
        for face_box in face_boxes:
            face = image.crop(face_box)
            face_encoding = encode_face(face)
            face_index = match_face(face_encoding)
            face_indexes.append(face_index)

        last_face_indexes = face_indexes
    else:
        face_indexes = last_face_indexes

    last_face_boxes = face_boxes

    recognized_faces = []
    for face_index, (left, top, right, bottom) in zip(face_indexes, face_boxes):
        location = (top * 4, right * 4, bottom * 4, left * 4)
        name = known_face_names[face_index]
        recognized_faces.append({
            'index': face_index,
            'location': location,
            'name': name
        })

    return recognized_faces
