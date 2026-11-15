"""
PyTorch Deep Learning Architectures (`src/adsb_rf/deep_models.py`).
Includes `RFAnomalyAutoencoder` (12 -> 24 -> 3 -> 24 -> 12) for unsupervised physical-layer
anomaly scoring and `IQConvNet1D` for raw complex baseband I/Q classification, with a deterministic
SVD autoencoder fallback when PyTorch is unavailable.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class RFAnomalyAutoencoder(nn.Module):
        """
        PyTorch Unsupervised Deep Autoencoder (`12 -> 24 -> 3 -> 24 -> 12`) for ADS-B RF
        & Kinematic Anomaly Detection.
        """

        def __init__(self, input_dim: int = 12, hidden_dim: int = 24, latent_dim: int = 3) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z

        def reconstruction_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """Computes per-frame Mean Squared Reconstruction Error ||x - x_hat||^2."""
            self.eval()
            with torch.no_grad():
                x_recon, _ = self.forward(x)
                mse = torch.mean((x - x_recon) ** 2, dim=1)
            return mse


    class IQConvNet1D(nn.Module):
        """
        PyTorch 1D Convolutional Neural Network (CNN) for End-to-End Raw I/Q RF Fingerprinting.
        """

        def __init__(self, num_classes: int = 3, embedding_dim: int = 24) -> None:
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels=2, out_channels=16, kernel_size=11, stride=2, padding=5),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(in_channels=32, out_channels=embedding_dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, iq_tensor: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            feats = self.feature_extractor(iq_tensor).squeeze(-1)
            logits = self.classifier(feats)
            return logits, feats


def complex_iq_to_tensor_array(frames: List[Dict[str, Any]], seq_len: int = 256) -> np.ndarray:
    """
    Converts each frame's `IQ_res` complex waveform into a normalized `(N, 2, seq_len)` float32 tensor.
    """
    n_frames = len(frames)
    tensor_arr = np.zeros((n_frames, 2, seq_len), dtype=np.float32)
    for i, f in enumerate(frames):
        iq = f.get("IQ_res") if f.get("IQ_res") is not None else f.get("IQ_resampled")
        if iq is None or len(iq) == 0:
            continue
        iq_slice = np.asarray(iq[:seq_len], dtype=np.complex128)
        L = len(iq_slice)
        ref_phase = np.angle(np.mean(iq_slice[:5])) if L >= 5 else 0.0
        iq_derot = iq_slice * np.exp(-1j * ref_phase)
        rms = float(np.sqrt(np.mean(np.abs(iq_derot) ** 2))) + 1e-12
        tensor_arr[i, 0, :L] = np.real(iq_derot) / rms
        tensor_arr[i, 1, :L] = np.imag(iq_derot) / rms
    return tensor_arr


def train_pytorch_models(
    train_frames: List[Dict[str, Any]],
    all_frames: List[Dict[str, Any]],
    X_train_scaled: np.ndarray,
    X_all_scaled: np.ndarray,
    y_train_idx: np.ndarray,
    num_classes: int,
    epochs: int = 18,
    lr: float = 1e-2,
) -> Dict[str, np.ndarray]:
    """
    Trains the PyTorch `RFAnomalyAutoencoder` and `IQConvNet1D` on the clean training frames.
    """
    if HAS_TORCH and len(train_frames) >= 3:
        torch.manual_seed(42)
        in_dim = X_train_scaled.shape[1]
        batch_sz = min(64, max(16, len(train_frames) // 20))

        ae_model = RFAnomalyAutoencoder(input_dim=in_dim, hidden_dim=24, latent_dim=3)
        ae_opt = torch.optim.AdamW(ae_model.parameters(), lr=lr, weight_decay=1e-4)
        mse_loss_fn = nn.MSELoss()

        x_train_t = torch.from_numpy(np.clip(X_train_scaled, -10.0, 10.0).astype(np.float32))
        x_all_t = torch.from_numpy(np.clip(X_all_scaled, -10.0, 10.0).astype(np.float32))
        ae_loader = DataLoader(TensorDataset(x_train_t), batch_size=batch_sz, shuffle=True)

        ae_model.train()
        for _ in range(epochs):
            for (batch_x,) in ae_loader:
                ae_opt.zero_grad()
                x_hat, _ = ae_model(batch_x)
                loss = mse_loss_fn(x_hat, batch_x)
                loss.backward()
                ae_opt.step()

        ae_mse_scores = ae_model.reconstruction_error(x_all_t).numpy()

        iq_train_np = complex_iq_to_tensor_array(train_frames, seq_len=256)
        iq_all_np = complex_iq_to_tensor_array(all_frames, seq_len=256)
        iq_train_t = torch.from_numpy(iq_train_np)
        iq_all_t = torch.from_numpy(iq_all_np)
        y_train_t = torch.from_numpy(y_train_idx.astype(np.int64))

        cnn_model = IQConvNet1D(num_classes=num_classes, embedding_dim=24)
        cnn_opt = torch.optim.AdamW(cnn_model.parameters(), lr=lr, weight_decay=1e-4)
        ce_loss_fn = nn.CrossEntropyLoss()

        cnn_loader = DataLoader(TensorDataset(iq_train_t, y_train_t), batch_size=batch_sz, shuffle=True)
        cnn_model.train()
        for _ in range(epochs):
            for batch_iq, batch_y in cnn_loader:
                cnn_opt.zero_grad()
                logits, _ = cnn_model(batch_iq)
                loss = ce_loss_fn(logits, batch_y)
                loss.backward()
                cnn_opt.step()

        cnn_model.eval()
        with torch.no_grad():
            all_logits, _ = cnn_model(iq_all_t)
            cnn_probs = torch.softmax(all_logits, dim=1).numpy()
            cnn_preds = np.argmax(cnn_probs, axis=1)

        return {
            "ae_mse_scores": ae_mse_scores,
            "cnn_probs": cnn_probs,
            "cnn_preds": cnn_preds,
        }

    X_tr_clip = np.clip(X_train_scaled, -10.0, 10.0)
    X_all_clip = np.clip(X_all_scaled, -10.0, 10.0)
    mean_tr = np.mean(X_tr_clip, axis=0)
    X_centered = X_tr_clip - mean_tr
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    k_bottleneck = min(3, vt.shape[0])
    w_enc = vt[:k_bottleneck, :]
    X_all_c = X_all_clip - mean_tr
    X_recon = np.dot(np.dot(X_all_c, w_enc.T), w_enc) + mean_tr
    ae_mse_scores = np.mean((X_all_clip - X_recon) ** 2, axis=1)

    return {
        "ae_mse_scores": ae_mse_scores,
        "cnn_probs": np.array([]),
        "cnn_preds": np.array([]),
    }
