# AudioAnomalyDetector/src/train/train_despawn.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_despawn_with_dataloader(
    model,
    train_tensor,
    epochs: int = 1000,
    loss_factor: float = 1.0,
    batch_size: int = 32,
    device: str = 'cuda'
):
    """
    EnhancedDeSpaWN 모델을 DataLoader로 학습하는 함수.

    Args:
        model (nn.Module): 학습할 EnhancedDeSpaWN 모델
        train_tensor (torch.Tensor): train data, shape=(N,1,L)
        epochs (int): 에포크 수
        loss_factor (float): 희소성 손실 가중치
        batch_size (int): 배치 크기
        device (str): 'cuda' 또는 'cpu'

    Returns:
        model: 학습된 모델
    """
    model = model.to(device)
    optimizer = optim.NAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

    dataset = torch.utils.data.TensorDataset(train_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_rec_loss = 0.0
        epoch_sparse_loss = 0.0

        pbar = tqdm(dataloader, desc=f"DespaWN Epoch {epoch}/{epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstruction, coeffs = model(batch, return_coeffs=True)

            rec_loss = torch.mean(torch.abs(batch - reconstruction))
            coeffs_flat = torch.cat([c.reshape(batch.size(0), -1) for c in coeffs], dim=1)
            sparse_loss = torch.mean(torch.abs(coeffs_flat))

            loss = rec_loss + loss_factor * sparse_loss
            loss.backward()
            optimizer.step()

            epoch_rec_loss += rec_loss.item() * batch.size(0)
            epoch_sparse_loss += sparse_loss.item() * batch.size(0)
            pbar.set_postfix({
                'rec_loss': f"{rec_loss.item():.4f}",
                'sparse_loss': f"{sparse_loss.item():.4f}",
                'total_loss': f"{loss.item():.4f}"  
            })

        n = len(dataset)
        print(f"Epoch {epoch}/{epochs} - Avg Rec: {epoch_rec_loss/n:.4f}, "
              f"Avg Sparse: {epoch_sparse_loss/n:.4f}")

    return model
