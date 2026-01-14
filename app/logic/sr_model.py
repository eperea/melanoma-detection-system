import torch
import torch.nn as nn
import numpy as np
import cv2

class SRCNN(nn.Module):
    def __init__(self, scale=2):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class SRPredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SRCNN(scale=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def reconstruct_from_patches(self, img, patch_size=64, stride=32, scale=2):
        """
        Reconstruye imagen SR a partir de parches.
        img: numpy array (H, W, 3) normalizado [0, 1] RGB
        """
        h, w, c = img.shape
        sr_h, sr_w = h * scale, w * scale
        sr_img = np.zeros((sr_h, sr_w, c), dtype=np.float32)
        weight = np.zeros((sr_h, sr_w, c), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patch = img[i:i+patch_size, j:j+patch_size, :]
                    patch_tensor = torch.tensor(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                    sr_patch = self.model(patch_tensor).cpu().squeeze(0).permute(1, 2, 0).numpy()
                    
                    si, sj = i*scale, j*scale
                    ps = sr_patch.shape[0] # Tamaño del parche SR
                    
                    sr_img[si:si+ps, sj:sj+ps, :] += sr_patch
                    weight[si:si+ps, sj:sj+ps, :] += 1

        # Rellenar bordes (donde weight es 0) con bicubic fallback
        mask = (weight == 0)
        if np.any(mask):
            fallback = cv2.resize(img, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
            sr_img[mask] = fallback[mask]
            weight[mask] = 1

        return np.clip(sr_img / weight, 0, 1)

    def predict(self, image_bytes):
        # Convertir bytes a imagen numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        
        # Inferencia
        sr_img = self.reconstruct_from_patches(img)
        
        # Convertir a formato 0-255 uint8 para guardar/mostrar
        sr_save = (sr_img * 255).astype(np.uint8)
        return sr_save
