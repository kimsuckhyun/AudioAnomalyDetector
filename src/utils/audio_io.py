# AudioAnomalyDetector/src/utils/audio_io.py

import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
from tqdm import tqdm

def _load_single_file(file_path: str, target_length: int = None):
    """
    단일 오디오 파일을 로드하고 정규화한 뒤, 필요에 따라 길이를 맞춥니다.
    """
    signal, sr = librosa.load(file_path, sr=None, mono=True)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    if target_length is not None:
        diff = target_length - signal.shape[0]
        if diff > 0:
            signal = np.pad(signal, (0, diff), mode='constant')
        else:
            signal = signal[:target_length]

    return signal, sr

def load_audio_batch(
    file_list: list[str],
    target_length: int = None,
    num_workers: int = None
) -> tuple[np.ndarray, list[int]]:
    """
    여러 오디오 파일을 병렬로 로드하여 넘파이 배열과 샘플레이트 리스트로 반환합니다.

    Args:
        file_list (list[str]): 로드할 파일 경로 리스트
        target_length (int, optional): 패딩/자르기 후 신호 길이
        num_workers (int, optional): 병렬 작업 수 (기본: CPU 코어 절반)

    Returns:
        signals (np.ndarray): (N, target_length) 또는 (N, 원본길이) 배열
        sample_rates (list[int]): 각 파일의 샘플레이트
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    signals = []
    sample_rates = []

    with ThreadPoolExecutor(max_workers=num_workers) as exec:
        futures = [exec.submit(_load_single_file, fp, target_length) for fp in file_list]
        for f in tqdm(futures, desc="Loading audio"):
            sig, sr = f.result()
            signals.append(sig)
            sample_rates.append(sr)

    return np.stack(signals), sample_rates
