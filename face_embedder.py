import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import torchvision.transforms as transforms

class FaceEmbedder:
    def __init__(self):
        self.model  = InceptionResnetV1(pretrained='vggface2').eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def get_embedding(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None

        rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor  = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def get_embedding_from_crop(self, crop_bgr):
        return self.get_embedding(crop_bgr)