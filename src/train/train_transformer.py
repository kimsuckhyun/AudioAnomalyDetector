# AudioAnomalyDetector/src/train/train_transformer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset import AudioDataset


def train_masked_transformer(
    model,
    train_tensor,
    epochs: int = 100,
    batch_size: int = 8,
    device: str = 'cuda'
):
    """
    EnhancedMaskedAudioTransformer 모델 학습 함수

    Args:
        model (nn.Module): 학습할 Transformer 모델
        train_tensor (torch.Tensor): train data, shape=(N,1,L)
        epochs (int): 에포크 수
        batch_size (int): 배치 크기
        device (str): 'cuda' 또는 'cpu'

    Returns:
        model: 학습된 모델
    """
    # 준비
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = torch.nn.MSELoss()

    # DataLoader
    dataset = AudioDataset(train_tensor.squeeze(1).numpy())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Transformer Epoch {epoch}/{epochs}")

        for batch in pbar:
            # batch: (batch_size, 1, L) -> (batch_size, 1, L)
            batch = batch.unsqueeze(-1) if batch.ndim == 3 else batch
            batch = batch.to(device)

            optimizer.zero_grad()
            recon_specs = model(batch)

            # 원본 스펙트로그램
            original_specs = model._compute_mel_spectrogram(batch)
            loss = mse_loss(recon_specs, original_specs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

    return model
