# AudioAnomalyDetector/src/main.py
import os
import torch
from src.utils.seed import set_seed
from src.utils.dataset import prepare_datasets
from src.models.despawn import EnhancedDeSpaWN
from src.models.transformer import EnhancedMaskedAudioTransformer
from src.models.vqvae import AudioAnomalyVQVAE
from src.train.train_despawn import train_despawn_with_dataloader
from src.train.train_transformer import train_masked_transformer
from src.train.train_vqvae import train_vqvae, visualize_vqvae_results


def main_enhanced_approach(
    base_folder: str,
    target_id: str = "id_00",
    l_train: int = 160000,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    전체 오디오 이상 탐지 파이프라인 실행
    """
    # 1) 재현성
    set_seed(seed)

    # 2) 데이터 준비
    train_tensor, test_tensor, test_labels, metadata, sr = prepare_datasets(
        base_folder, target_id, l_train
    )
    print(f"Train shape: {train_tensor.shape}, Test shape: {test_tensor.shape}, SR: {sr}")

    # 3) 모델 초기화
    despawn_model = EnhancedDeSpaWN(
        kernel_init=[-0.0105974,0.0328830,0.0308413,-0.1870348,-0.0279837,0.6308807,0.7148465,0.2303778],
        kern_trainable=True,
        level=17,
        kernels_constraint='PerLayer',
        init_ht=0.3,
        train_ht=True
    )
    transformer_model = EnhancedMaskedAudioTransformer(
        n_mels=64, hop_length=512, n_fft=2048, max_len=l_train,
        patch_size=16, embed_dim=768, depth=12, heads=12, mask_ratio=0.75
    )
    vqvae_model = AudioAnomalyVQVAE(
        despawn_model, transformer_model,
        fusion_dim=256, latent_dim=128,
        num_embeddings=512, commitment_cost=0.25
    )

    # 4) 학습
    despawn_model = train_despawn_with_dataloader(
        despawn_model, train_tensor,
        epochs=100, loss_factor=1.0,
        batch_size=1, device=device
    )
    transformer_model = train_masked_transformer(
        transformer_model, train_tensor,
        epochs=50, batch_size=1, device=device
    )
    vqvae_model, best_auc, history = train_vqvae(
        vqvae_model, train_tensor, test_tensor, test_labels,
        epochs=100, batch_size=1, device=device,
        save_path=os.path.join(os.getcwd(), 'best_vqvae.pth')
    )

    # 5) 평가 및 시각화
    results = visualize_vqvae_results(
        vqvae_model, test_tensor, test_labels,
        device=device, use_tta=True, save_path='vqvae_results.png'
    )

    # 6) 모델 저장
    save_dict = {
        'despawn': despawn_model.state_dict(),
        'transformer': transformer_model.state_dict(),
        'vqvae': vqvae_model.state_dict(),
        'metadata': metadata,
        'results': results
    }
    torch.save(save_dict, 'final_audio_anomaly_model.pth')
    print("Pipeline complete. Models and results saved.")

# Entry point
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Base data folder')
    parser.add_argument('--id', type=str, default='id_00')
    parser.add_argument('--length', type=int, default=160000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main_enhanced_approach(args.data, args.id, args.length, device=args.device)