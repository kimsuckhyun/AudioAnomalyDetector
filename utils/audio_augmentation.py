import numpy as np
import librosa

def augment_audio(signal, sr):
    """
    오디오 데이터 증강 함수

    여러 증강 기법을 적용하여 데이터 다양성을 확보하는 함수입니다.

    Parameters:
        signal (np.ndarray): 입력 오디오 신호.
        sr (int): 샘플링 레이트.

    Returns:
        list: 증강된 오디오 신호 리스트.
    """
    augmented = []
    
    # 1. 시간 이동 (Time Shift)
    shift_factor = np.random.uniform(-0.1, 0.1)
    shift_amount = int(len(signal) * shift_factor)
    shifted_signal = np.roll(signal, shift_amount)
    augmented.append(shifted_signal)
    
    # 2. 배경 노이즈 추가 (Background Noise Addition)
    noise_factor = np.random.uniform(0.001, 0.005)
    noise = np.random.normal(0, noise_factor, len(signal))
    noisy_signal = signal + noise
    augmented.append(noisy_signal)
    
    # 3. 피치 변화 (Pitch Shift)
    pitch_factor = np.random.uniform(-2, 2)
    pitched_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_factor)
    augmented.append(pitched_signal)
    
    # 4. 타임 스트레칭 (Time Stretch)
    stretch_factor = np.random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(signal, rate=stretch_factor)
    if len(stretched) < len(signal):
        stretched = np.pad(stretched, (0, len(signal) - len(stretched)), mode='constant')
    else:
        stretched = stretched[:len(signal)]
    augmented.append(stretched)
    
    # 5. 랜덤 볼륨 조정 (Random Volume Adjustment)
    volume_factor = np.random.uniform(0.8, 1.2)
    volume_adjusted = signal * volume_factor
    augmented.append(volume_adjusted)
    
    return augmented
