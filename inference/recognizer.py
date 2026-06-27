"""
Module for running Optical Character Recognition (OCR) inference using the CRNN model.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image

from inference.crnn_network import CRNN, CTCLabelConverter 

class CRNNRecognizer:
    """
    A wrapper class for the CRNN character recognition model.

    Attributes:
        device (torch.device): The computation device ('cuda' or 'cpu').
        converter (CTCLabelConverter): The text-to-index and index-to-text converter.
        model (CRNN): The CRNN neural network model instance.
        transform (torchvision.transforms.Compose): Image preprocessing pipeline.
    """

    def __init__(self, model_path, charset):
        """
        Initializes the CRNNRecognizer model and loads the pre-trained weights.

        Args:
            model_path (str): The file path to the trained CRNN model weights (.pth).
            charset (str): The string representing all possible recognizable characters.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.converter = CTCLabelConverter(charset)
        self.model = CRNN(
            vocab_size=self.converter.get_vocab_size(),
            hidden_size=256
        )

        # Memuat bobot model
        # # TODO: [Saran perbaikan] Gunakan weights_only=True pada torch.load untuk alasan keamanan (menghindari arbitrary code execution dari malicious pickle file).
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Memuat weights dari key: 'model_state_dict'")
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Memuat weights secara langsung")
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()

        # Transformasi input gambar agar sesuai dengan format yang diharapkan model
        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def recognize(self, img_bgr):
        """
        Predicts the text present in the provided image array.

        Args:
            img_bgr (numpy.ndarray): The cropped image containing the text, in BGR format.

        Returns:
            str: The recognized text string.
        """
        # Konversi BGR ke RGB (karena model dilatih dengan gambar RGB/PIL)
        img_rgb = img_bgr[:, :, ::-1]
        img_pil = Image.fromarray(img_rgb)

        img_tensor = self.transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Inferensi tanpa perhitungan gradient untuk efisiensi memori dan waktu komputasi
        with torch.no_grad():
            outputs = self.model(img_tensor)
            preds = torch.argmax(outputs, dim=2)
            text = self.converter.decode(preds[0].cpu().numpy())

        return text
