import numpy as np


def is_face_forward(landmarks):
    """
    判断人脸是否正对相机，并考虑人脸的俯仰角。

    参数:
    - landmarks: 人脸的关键点坐标，假设是一个包含五个关键点（两只眼睛、鼻尖、两只嘴角）的列表。

    返回:
    - is_forward: 布尔值，表示人脸是否正对相机，没有显著的俯仰、偏航或滚动角。
    """
    # 提取两只眼睛和鼻尖的坐标
    eye_left = np.array(landmarks[0])
    eye_right = np.array(landmarks[1])
    nose = np.array(landmarks[2])

    # 计算眼睛水平和垂直方向的差异
    horizontal_diff = abs(eye_left[1] - eye_right[1])
    vertical_diff = abs(eye_left[0] - eye_right[0])

    # 计算眼睛中心点
    eye_center = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)

    # 计算鼻尖到眼睛中心线的垂直距离
    vertical_nose_diff = abs(nose[1] - eye_center[1])

    # 计算眼睛中心到鼻尖的直线距离，以估计俯仰角
    pitch_diff = np.linalg.norm(nose - eye_center)

    # 姿态判断逻辑，这里可以根据实际情况调整阈值
    if horizontal_diff < 0.1 * vertical_diff and vertical_nose_diff < 0.1 * vertical_diff and pitch_diff < 0.2 * vertical_diff:
        return True
    else:
        return False


def evaluate_face_quality(box, prob, image, landmarks, quality_threshold=0.5):
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

    # 加入姿态评分（基于眼睛和鼻子的位置简单估计头部是否正对摄像头）
    eye_left, eye_right, nose = landmarks[:3]
    dx = eye_right[0] - eye_left[0]
    dy = eye_right[1] - eye_left[1]
    eye_dist = np.sqrt(dx ** 2 + dy ** 2)
    nose_eye_center_dist = np.sqrt((nose[0] - (eye_left[0] + eye_right[0]) / 2) ** 2 +
                                   (nose[1] - (eye_left[1] + eye_right[1]) / 2) ** 2)
    pose_score = 1 - min(nose_eye_center_dist / eye_dist, 1)  # 简化版的姿态评分

    # 综合评分
    quality_score = size_score * confidence_score * pose_score

    if quality_score > quality_threshold:
        return True
    else:
        return False
