import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def encode_face(face_image):
    """Encode a face image into a high-dimensional vector."""
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])
    face_tensor = preprocess(face_image).unsqueeze(0)
    with torch.no_grad():
        face_encoding = resnet(face_tensor)
    return face_encoding.detach().cpu().numpy()[0]
