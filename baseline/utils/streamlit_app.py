import os
import glob
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 로컬 오디오 데이터가 있는 디렉토리 (변경 가능)
DATA_DIR = "/home/sh/Sound_Dataset"

def find_wav_files(root_dir):
    """
    주어진 디렉토리(root_dir) 내에서 모든 .wav 파일을 재귀적으로 검색하여
    절대 경로 리스트로 반환합니다.
    """
    pattern = os.path.join(root_dir, "**", "*.wav")
    file_list = glob.glob(pattern, recursive=True)
    return file_list

def load_audio_file(file_path, sr=None):
    """
    지정한 경로의 오디오 파일을 librosa를 사용하여 로드합니다.
    """
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    return fig

def plot_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title="Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig

def main():
    st.title("로컬 오디오 시각화")
    st.write("디렉토리 내에서 WAV 파일을 재귀적으로 검색하여 선택한 파일의 파형과 Mel Spectrogram을 확인합니다.")
    
    # DATA_DIR 내의 모든 WAV 파일을 재귀적으로 검색합니다.
    wav_files = find_wav_files(DATA_DIR)
    if not wav_files:
        st.error(f"디렉토리 {DATA_DIR} 내에 WAV 파일이 없습니다.")
        return

    # 파일 경로를 보기 좋게 표시하기 위해 상대 경로 또는 파일명만 표시할 수 있습니다.
    # 여기서는 전체 경로 대신 파일명을 표시합니다.
    file_names = [os.path.relpath(path, DATA_DIR) for path in wav_files]
    selected_idx = st.selectbox("오디오 파일 선택", range(len(file_names)), format_func=lambda i: file_names[i])
    selected_file = wav_files[selected_idx]
    
    st.write("선택한 파일:", selected_file)
    
    # 오디오 파일 로드 및 재생
    y, sr = load_audio_file(selected_file)
    st.write("Sampling rate:", sr)
    
    with open(selected_file, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")
    
    st.subheader("Waveform")
    fig_wave = plot_waveform(y, sr)
    st.pyplot(fig_wave)
    
    st.subheader("Mel Spectrogram")
    fig_spec = plot_mel_spectrogram(y, sr)
    st.pyplot(fig_spec)

if __name__ == "__main__":
    main()
