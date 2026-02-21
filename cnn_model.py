import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


CLASS_NAMES: List[str] = [
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

CLASS_TO_FEN: Dict[str, str] = {
    "empty": ".",
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k",
}


class ChessPieceCNN(nn.Module):
    """Small CNN for grayscale square classification."""

    def __init__(self, num_classes: int = 13) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SquarePreprocessor:
    """Image preprocessing aligned between training and inference."""

    def __init__(self, image_size: int = 64, mean: float = 0.5, std: float = 0.5) -> None:
        self.image_size = int(image_size)
        self.mean = float(mean)
        self.std = float(std)

    def preprocess_batch(self, squares_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        processed = []
        for square in squares_bgr:
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            x = gray.astype(np.float32) / 255.0
            x = (x - self.mean) / max(self.std, 1e-6)
            processed.append(x)
        arr = np.stack(processed, axis=0)[:, None, :, :]
        return torch.from_numpy(arr).float()


class ChessPieceClassifier:
    """Loads a trained checkpoint and predicts 64 squares in one batch."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.class_names: List[str] = list(CLASS_NAMES)
        self.preprocessor = SquarePreprocessor()
        self.model = ChessPieceCNN(num_classes=len(self.class_names)).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Train model first with train_cnn.py."
            )
        blob = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(blob, dict) and "state_dict" in blob:
            state_dict = blob["state_dict"]
            if isinstance(blob.get("class_names"), list):
                self.class_names = blob["class_names"]
            if "image_size" in blob:
                self.preprocessor.image_size = int(blob["image_size"])
            if "mean" in blob:
                self.preprocessor.mean = float(blob["mean"])
            if "std" in blob:
                self.preprocessor.std = float(blob["std"])
        elif isinstance(blob, dict):
            state_dict = blob
        else:
            raise ValueError("Unsupported checkpoint format.")

        self.model = ChessPieceCNN(num_classes=len(self.class_names)).to(self.device)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v
        self.model.load_state_dict(cleaned, strict=False)

    @torch.inference_mode()
    def predict_64(self, squares_bgr: Sequence[np.ndarray]) -> Tuple[List[str], List[float]]:
        if len(squares_bgr) != 64:
            raise ValueError(f"predict_64 expects 64 squares, got {len(squares_bgr)}.")
        batch = self.preprocessor.preprocess_batch(squares_bgr).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        labels = [self.class_names[int(i)] for i in idx.cpu().numpy()]
        confs = [float(c) for c in conf.cpu().numpy()]
        return labels, confs
