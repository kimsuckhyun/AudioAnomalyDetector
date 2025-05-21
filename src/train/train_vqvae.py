# AudioAnomalyDetector/src/train/train_vqvae.py
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


def test_time_augmentation(model, audio, num_augmentations: int = 5):
    device = next(model.parameters()).device
    scores = [model.compute_anomaly_score(audio.to(device))]
    for _ in range(num_augmentations):
        aug = audio + torch.randn_like(audio) * np.random.uniform(0.001, 0.002)
        scores.append(model.compute_anomaly_score(aug.to(device)))
    return torch.mean(torch.cat(scores, dim=1), dim=1, keepdim=True)


def evaluate_vqvae_anomaly_detector(model, dataloader, device: str = 'cuda', use_tta: bool = False):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            if use_tta:
                sc = test_time_augmentation(model, inputs)
            else:
                sc = model.compute_anomaly_score(inputs)
            sc = torch.nan_to_num(sc, nan=0.0)
            all_scores.extend(sc.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    if len(all_scores) == 0 or len(np.unique(all_labels))<=1 or len(np.unique(all_scores))<=1:
        return 0.5
    return roc_auc_score(all_labels, all_scores)


def train_vqvae(
    model,
    train_tensor,
    test_tensor,
    test_labels,
    epochs: int = 100,
    batch_size: int = 8,
    device: str = 'cuda',
    save_path: str = 'best_vqvae_anomaly_detector.pth',
    use_tta: bool = False
):
    device = torch.device(device)
    model = model.to(device)

    train_ds = TensorDataset(train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_ds = TensorDataset(test_tensor, torch.tensor(test_labels, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    best_auc = 0.0
    history = {'roc_auc': [], 'rec_loss': [], 'vq_loss': [], 'total_loss': [], 'lr': []}

    for epoch in range(1, epochs+1):
        model.train()
        epoch_rec, epoch_vq, epoch_tot = 0.0, 0.0, 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"VQ-VAE Epoch {epoch}/{epochs}")
        for (batch,), in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, fused, vq_loss, indices = None, None, None, None
            # forward
            quantized, vq_loss, fused, _ = model.encode(
                model.despawn.extract_features(batch),
                model.trans.extract_features(batch)
            )
            reconstructed = model.decode(quantized)
            rec_loss = F.mse_loss(reconstructed, fused)
            loss = rec_loss + 0.1 * vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_rec += rec_loss.item() * batch.size(0)
            epoch_vq += vq_loss.item() * batch.size(0)
            epoch_tot += loss.item() * batch.size(0)
            count += batch.size(0)
            pbar.set_postfix({'rec':f"{rec_loss.item():.4f}",'vq':f"{vq_loss.item():.4f}",'tot':f"{loss.item():.4f}"})
        scheduler.step()
        roc_auc = evaluate_vqvae_anomaly_detector(model, test_loader, device, use_tta)
        history['roc_auc'].append(roc_auc)
        history['rec_loss'].append(epoch_rec/count)
        history['vq_loss'].append(epoch_vq/count)
        history['total_loss'].append(epoch_tot/count)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch}: AUC={roc_auc:.4f}, Rec={epoch_rec/count:.4f}, VQ={epoch_vq/count:.4f}")

        if roc_auc > best_auc:
            best_auc = roc_auc
            torch.save({'model_state': model.state_dict(), 'auc': best_auc, 'epoch': epoch}, save_path)
            print(f"Saved best model at epoch {epoch}, AUC={best_auc:.4f}")
    # Load best
    ckpt = torch.load(save_path)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded best model: epoch {ckpt['epoch']}, AUC={ckpt['auc']:.4f}")
    return model, best_auc, history


def visualize_vqvae_results(
    model,
    test_tensor,
    test_labels,
    device: str = 'cuda',
    use_tta: bool = True,
    save_path: str = 'vqvae_results.png'
):
    device = torch.device(device)
    model = model.to(device)
    ds = TensorDataset(test_tensor, torch.tensor(test_labels, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    all_scores, all_latent, all_idx = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            if use_tta:
                sc = test_time_augmentation(model, inputs)
            else:
                sc = model.compute_anomaly_score(inputs)
            rec, fused, quantized, vq_loss, idx = model(inputs, return_latent=True)
            all_scores.extend(sc.cpu().numpy().flatten())
            all_latent.append(quantized.cpu().numpy())
            all_idx.append(idx.cpu().numpy())
    all_latent = np.vstack(all_latent)
    all_idx = np.concatenate(all_idx)

    # t-SNE
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, random_state=42).fit_transform(all_latent)

    # plot
    plt.figure(figsize=(12,8))
    plt.scatter(emb[:,0], emb[:,1], c=all_scores, cmap='viridis', s=5)
    plt.colorbar(label='Anomaly Score')
    plt.title('t-SNE of Latent Representations')
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    return {'tsne': emb, 'scores': all_scores, 'indices': all_idx}
