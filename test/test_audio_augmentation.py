import numpy as np
import librosa
from ..utils.audio_augmentation import augment_audio

def main():
    # 예제 오디오 파일 로딩
    audio_path = '../sample_data/-6dB_fan/fan/id_00/abnormal/00000000.wav'  # 사용하고자 하는 오디오 파일 경로 지정
    signal, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # 입력 신호 정규화 (옵션)
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # 증강 함수 적용
    augmented_signals = augment_audio(signal, sr)
    
    # 결과 출력: 원본 및 증강된 샘플의 기본 통계 정보 출력
    print("원본 신호 길이:", len(signal))
    print("증강된 샘플 개수:", len(augmented_signals))
    for i, aug in enumerate(augmented_signals):
        print(f"증강 샘플 {i+1}: 길이 {len(aug)}, 평균 {np.mean(aug):.4f}, 표준편차 {np.std(aug):.4f}")

if __name__ == '__main__':
    main()
