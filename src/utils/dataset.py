# AudioAnomalyDetector/src/utils/dataset.py

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_io import load_audio_batch


def prepare_datasets(
    base_folder: str,
    target_id: str,
    l_train: int,
    num_workers: int = None
):
    """
    주어진 폴더에서 특정 ID의 정상/비정상 오디오를 로드하여 학습 및 테스트 데이터셋을 준비합니다.

    Args:
        base_folder (str): 데이터셋이 위치한 최상위 폴더 경로
        target_id (str): 사용할 ID 폴더 이름 (예: "id_00")
        l_train (int): 패딩/자르기 후 신호 길이
        num_workers (int, optional): 오디오 로딩에 사용할 병렬 작업 수

    Returns:
        train_tensor (Tensor): 학습용 오디오 텐서, shape=(N_train, 1, L)
        test_tensor (Tensor): 테스트용 오디오 텐서, shape=(N_test, 1, L)
        test_labels (ndarray): 테스트 라벨 (0=정상, 1=비정상)
        metadata (dict): 파일 경로 및 메타데이터 정보
        sample_rate (int): 오디오 샘플레이트
    """
    # ID 폴더 경로 검증
    id_folder = os.path.join(base_folder, target_id)
    if not os.path.isdir(id_folder):
        raise FileNotFoundError(f"ID 폴더가 없습니다: {id_folder}")

    # 정상/비정상 폴더
    normal_folder = os.path.join(id_folder, "normal")
    abnormal_folder = os.path.join(id_folder, "abnormal")

    if not os.path.isdir(normal_folder):
        raise FileNotFoundError(f"정상 폴더가 없습니다: {normal_folder}")

    # 파일 목록 수집
    normal_files = [os.path.join(normal_folder, f) for f in os.listdir(normal_folder)
                    if f.lower().endswith('.wav')]
    abnormal_files = []
    if os.path.isdir(abnormal_folder):
        abnormal_files = [os.path.join(abnormal_folder, f) for f in os.listdir(abnormal_folder)
                          if f.lower().endswith('.wav')]

    if len(normal_files) == 0:
        raise ValueError(f"정상 파일이 없습니다: {normal_folder}")

    # 테스트용 정상 파일 개수 결정
    random.shuffle(normal_files)
    n_ab = len(abnormal_files)
    n_test_norm = min(n_ab, len(normal_files) // 2)

    test_normal_files = normal_files[:n_test_norm]
    train_files = normal_files[n_test_norm:]
    test_abnormal_files = abnormal_files

    # 메타데이터 기록
    metadata = {
        'train': [(f, target_id) for f in train_files],
        'test': [(f, target_id, 0) for f in test_normal_files]
                + [(f, target_id, 1) for f in test_abnormal_files]
    }

    # 오디오 로드
    train_signals, sr_list = load_audio_batch(train_files, target_length=l_train, num_workers=num_workers)
    test_norm_signals, _ = load_audio_batch(test_normal_files, target_length=l_train, num_workers=num_workers)
    test_abn_signals, _ = load_audio_batch(test_abnormal_files, target_length=l_train, num_workers=num_workers)

    # 테스트 데이터 및 라벨
    test_signals = np.concatenate([test_norm_signals, test_abn_signals], axis=0)
    test_labels = np.concatenate([np.zeros(len(test_norm_signals)), np.ones(len(test_abn_signals))])

    # 텐서 변환 (shape: N x 1 x L)
    train_tensor = torch.from_numpy(train_signals).float().unsqueeze(1)
    test_tensor = torch.from_numpy(test_signals).float().unsqueeze(1)

    # 대표 샘플레이트
    sample_rate = sr_list[0] if sr_list else None

    return train_tensor, test_tensor, test_labels, metadata, sample_rate


class AudioDataset(Dataset):
    """
    오디오 텐서를 래핑하는 PyTorch Dataset
    """
    def __init__(self, signals: torch.Tensor):
        """
        Args:
            signals (Tensor): shape=(N, 1, L)
        """
        self.signals = signals

    def __len__(self):
        return self.signals.size(0)

    def __getitem__(self, idx: int):
        return self.signals[idx]
