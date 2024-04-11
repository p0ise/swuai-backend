import numpy as np

def is_face_forward(landmarks):
    """
    判断人脸是否正对相机,并考虑人脸的俯仰角。

    参数:
    - landmarks: 人脸的关键点坐标,假设是一个包含五个关键点(两只眼睛、鼻尖、两只嘴角)的列表。

    返回:
    - is_forward: 布尔值,表示人脸是否正对相机,没有显著的俯仰、偏航或滚动角。
    """
    # 提取两只眼睛和鼻尖的坐标
    eye_left = np.array(landmarks[0])
    eye_right = np.array(landmarks[1])
    nose = np.array(landmarks[2])
    mouth_left = np.array(landmarks[3])
    mouth_right = np.array(landmarks[4])

    # 计算中点
    eye_center = (eye_left + eye_right) / 2

    # 计算方向向量
    direction_vector = eye_right - eye_left

    # 计算单位向量
    unit_vector = direction_vector / np.linalg.norm(direction_vector)

    # 计算鼻子到左右眼连线的向量
    nose_to_eye_line_vector = nose - eye_left

    # 计算投影长度
    projection_length = np.dot(nose_to_eye_line_vector, unit_vector)

    # 计算垂足坐标
    foot_of_perpendicular = eye_left + projection_length * unit_vector

    # 计算垂足到中点的距离
    foot_offset = np.linalg.norm(foot_of_perpendicular - eye_center)
    eye_dist = np.linalg.norm(eye_right - eye_left)

    # 设定偏移量阈值,判断是否为正面
    threshold = 0.1 * eye_dist
    if foot_offset < threshold:
        return True
    else:
        return False

def evaluate_face_quality(box, prob, image, landmarks, quality_threshold=0.8):
    """
    根据人脸的大小、置信度和其他因素来评估人脸的质量。
    返回是否通过质量检查。

    参数:
    - box: 人脸框（x1, y1, x2, y2）
    - prob: 人脸检测的置信度
    - image: 包含人脸的图像（PIL.Image格式）
    - landmarks: 人脸的关键点坐标
    - quality_threshold: 质量评分的阈值，高于此阈值表示质量足够

    返回:
    - boolean: 表示人脸是否通过质量检查
    """
    # 基于人脸大小和置信度的简单评分：0-1
    face_area = (box[2] - box[0]) * (box[3] - box[1])
    size_score = np.sqrt(face_area) / 160 if np.sqrt(face_area) / 160 <= 1 else 1

    # 置信度分数，prob在0到1之间
    confidence_score = prob

    # 综合评分
    quality_score = size_score * confidence_score

    if quality_score > quality_threshold:
        return True
    else:
        return False
