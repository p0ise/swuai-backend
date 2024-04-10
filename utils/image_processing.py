import base64

import cv2
import numpy as np
from PIL import Image


def cv2_to_pil(image):
    """Convert a CV2 image to PIL format."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def parse_frame_data(frame_data):
    """Parse frame data from base64 encoding."""
    frame_data = base64.b64decode(frame_data.split(',')[1])
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame, flags=1)
    return cv2_to_pil(frame)
