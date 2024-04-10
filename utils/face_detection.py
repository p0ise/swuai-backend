import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化MTCNN和InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)


def find_faces(image):
    # 使用MTCNN检测人脸，同时返回框、置信度和关键点
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    if boxes is not None:
        return boxes, probs, landmarks
    else:
        return [], [], []


def find_primary_face(image):
    """
    查找图像中最显著的人脸。

    参数:
    - image: PIL.Image对象，包含待检测的图像。

    返回:
    - primary_face: 最显著的人脸信息，包括框、置信度和关键点。
                     如果没有检测到人脸，则返回None。
    """
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        # 选择置信度最高的人脸作为最显著的人脸
        max_prob_index = np.argmax(probs)
        primary_box = boxes[max_prob_index]
        primary_prob = probs[max_prob_index]
        primary_landmark = landmarks[max_prob_index]
        return primary_box, primary_prob, primary_landmark
    else:
        return None


def align_face(image, box, landmark):
    # 计算两眼之间的角度
    eye_left = np.array(landmark[0])
    eye_right = np.array(landmark[1])
    d_y = eye_right[1] - eye_left[1]
    d_x = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(d_y, d_x)) - 180

    # 应用仿射变换对齐人脸
    desired_width = 160
    desired_height = 160
    desired_dist = desired_width * 0.5
    eyes_center = ((eye_left[0] + eye_right[0]) // 2, (eye_left[1] + eye_right[1]) // 2)
    scale = 1
    m = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    # 更新仿射变换矩阵的平移量
    m[0, 2] += desired_dist - eyes_center[0]
    m[1, 2] += desired_height * 0.4 - eyes_center[1]

    # 应用仿射变换
    aligned = cv2.warpAffine(np.array(image), m, (desired_width, desired_height), flags=cv2.INTER_CUBIC)

    return Image.fromarray(aligned)
