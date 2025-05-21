# utils/visualization.py
import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_waveform(y, sr):
    """
    주어진 오디오 신호(y)와 샘플링 레이트(sr)를 사용하여 파형을 그립니다.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    return fig

def plot_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    주어진 오디오 신호(y)와 샘플링 레이트(sr)를 사용하여 Mel Spectrogram을 계산하고 시각화합니다.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title="Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig

def load_audio_file(audio_path, sr=None, mono=True):
    """
    지정한 경로의 오디오 파일을 로드합니다.
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=mono)
    return y, sr

def streamlit_audio_visualization(uploaded_file):
    """
    Streamlit 앱에서 업로드된 오디오 파일을 받아서 원본 파형과 Mel Spectrogram을 시각화하는 함수.
    """
    # 오디오 재생
    st.audio(uploaded_file, format="audio/wav")
    
    # 업로드된 파일을 임시 파일로 저장 후 로드
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    y, sr = load_audio_file(tmp_path)
    
    st.subheader("Waveform")
    fig_wave = plot_waveform(y, sr)
    st.pyplot(fig_wave)
    
    st.subheader("Mel Spectrogram")
    fig_spec = plot_mel_spectrogram(y, sr)
    st.pyplot(fig_spec)
    
    os.remove(tmp_path)
