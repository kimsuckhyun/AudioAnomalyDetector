# aad/utils/feature.py
import os
import sys
import glob
import numpy as np
import librosa         # 추가: librosa import
from tqdm import tqdm
from .io import file_load, demux_wav

def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0, logger=None):
    dims = n_mels * frames
    sr, y = demux_wav(file_name, logger=logger)
    if sr is None or y is None:
        if logger:
            logger.warning("Skipping file: %s", file_name)
        return np.empty((0, dims), float)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length,
                                               n_mels=n_mels, power=power)
    log_mel_spec = 20.0 / power * np.log10(mel_spec + sys.float_info.epsilon)
    vectorarray_size = log_mel_spec.shape[1] - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)
    vector_array = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vector_array[:, n_mels * t : n_mels * (t+1)] = log_mel_spec[:, t:t+vectorarray_size].T
    return vector_array

def list_to_vector_array(file_list, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0, logger=None):
    dims = n_mels * frames
    dataset = None
    for idx, file in enumerate(tqdm(file_list, desc="Processing files", leave=False)):
        vec_arr = file_to_vector_array(file, n_mels, frames, n_fft, hop_length, power, logger=logger)
        if idx == 0:
            dataset = np.zeros((vec_arr.shape[0] * len(file_list), dims), float)
        start = vec_arr.shape[0] * idx
        end = vec_arr.shape[0] * (idx + 1)
        dataset[start:end, :] = vec_arr
    return dataset

def dataset_generator(target_dir, normal_dir_name="normal", abnormal_dir_name="abnormal", ext="wav", logger=None):
    normal_files = sorted(glob.glob(os.path.join(target_dir, normal_dir_name, f"*.{ext}")))
    normal_labels = np.zeros(len(normal_files))
    if len(normal_files) == 0:
        if logger:
            logger.exception("No normal data in %s", target_dir)
        raise ValueError("No normal data")
    abnormal_files = sorted(glob.glob(os.path.join(target_dir, abnormal_dir_name, f"*.{ext}")))
    abnormal_labels = np.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        if logger:
            logger.exception("No abnormal data in %s", target_dir)
        raise ValueError("No abnormal data")
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = normal_files[:len(abnormal_files)] + abnormal_files
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels))
    if logger:
        logger.info("Train files: %d, Eval files: %d", len(train_files), len(eval_files))
    return train_files, train_labels, eval_files, eval_labels
