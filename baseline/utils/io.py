import os
import pickle
import glob
import logging
import librosa

def save_pickle(filename, data, logger=None):
    if logger:
        logger.info("Saving pickle: %s", filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename, logger=None):
    if logger:
        logger.info("Loading pickle: %s", filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def file_load(wav_name, mono=False, logger=None):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except Exception as e:
        if logger:
            logger.error("File broken or not exists: %s. Error: %s", wav_name, str(e))
        return None, None

def demux_wav(wav_name, channel=0, logger=None):
    multi_channel_data, sr = file_load(wav_name, mono=False, logger=logger)
    if multi_channel_data is None:
        return None, None
    if multi_channel_data.ndim <= 1:
        return sr, multi_channel_data
    return sr, multi_channel_data[channel, :]