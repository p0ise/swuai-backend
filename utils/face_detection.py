import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    if boxes is not None:
        return boxes[0], probs[0], landmarks[0]
    else:
        return None


def align_face(image, box, landmark):
    # 计算两眼之间的角度
    eye_left = np.array(landmark[0])
    eye_right = np.array(landmark[1])
    d_y = eye_right[1] - eye_left[1]
    d_x = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(d_y, d_x))

    # 计算box中心作为旋转中心
    box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    # 将PIL.Image转换为numpy数组以便使用cv2函数处理
    image_np = np.array(image)

    # 应用仿射变换对齐整个画面
    scale = 1
    M = cv2.getRotationMatrix2D(box_center, angle, scale)
    rotated_image_np = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]))

    # 将旋转后的numpy数组图像转换回PIL.Image
    rotated_image = Image.fromarray(rotated_image_np)

    # 从旋转后的图像中截取人脸区域
    face_image = rotated_image.crop(box)

    # 缩放图像到指定的尺寸
    aligned_face = face_image.resize((160, 160))

    return aligned_face
