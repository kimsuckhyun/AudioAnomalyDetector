# AudioAnomalyDetector

An end-to-end solution for detecting anomalies in industrial machine sounds. This project utilizes advanced machine learning methods to analyze acoustic signals, aiming to improve fault detection and maintenance efficiency.

이 프로젝트는 오디오 이상 탐지(Anomaly Detection)를 위한 모델 학습 및 평가 코드를 포함하고 있습니다.  
주요 기능은 다음과 같습니다:

- **Pretrain**:  
  여러 dB 레벨 및 여러 장치(예: fan, valve, slider, pump)와 ID별로 오토인코더를 학습하여 모델 가중치를 저장합니다.
  
- **Transfer**:  
  대상 디렉토리(예: DB/장치/ID 폴더)의 정상/비정상 데이터를 기반으로, 다른 장치의 pretrain 모델을 불러와 transfer 학습 및 평가(Roc AUC 산출)를 수행합니다.
  
- **Non-transfer**:  
  대상 디렉토리의 정상과 비정상 데이터를 모두 사용하여 오토인코더를 학습하고, 평가(Roc AUC 산출)를 수행합니다.
  
- **Visualization (Streamlit 앱)**:  
  로컬 오디오 파일을 재귀적으로 검색하여 선택하면 원본 파형과 Mel Spectrogram을 시각화합니다.

## 폴더 구조

프로젝트의 주요 폴더 구조는 아래와 같습니다:

```
AudioAnomalyDetector/
├── baseline/
│   ├── __init__.py
│   ├── pretrain/
│   │   ├── __init__.py
│   │   └── make_pretrain.py
│   ├── transfer/
│   │   ├── __init__.py
│   │   └── transfer_pretrain.py
│   ├── non_transfer/
│   │   ├── __init__.py
│   │   └── non_transfer_learning.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── feature.py
│       ├── logging_setup.py
│       ├── model.py
│       └── visualization.py
├── config/
│   ├── make_pretrain_v1.yaml
│   ├── transfer_pretrain_v1.yaml
│   └── non_transfer_learning.yaml
├── sample_data/                # 오디오 데이터 폴더 (예: -6dB_fan, 0dB_fan, 6dB_pump 등)
└── README.md
```

## 실행 방법

### 1. 환경 설정

먼저, `requirements.txt`에 있는 패키지들을 설치합니다:

```bash
pip install -r requirements.txt
```

### 2. Pretrain 실행

Pretrain 학습은 `baseline/pretrain/make_pretrain.py`를 실행하여,  
base_directory 아래의 각 dB 레벨×장치×ID별로 정상 데이터(예: `normal` 폴더 내 WAV 파일)를 기반으로 오토인코더를 학습합니다.

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다:

```bash
python -m baseline.pretrain.make_pretrain
```

### 3. Transfer 실행

Transfer 학습 및 평가는 `baseline/transfer/transfer_pretrain.py`를 실행합니다.  
이 스크립트는 대상 디렉토리 (예: `DB/장치/ID` 구조)의 데이터를 이용해 transfer 학습을 수행하고, 평가 결과(Roc AUC)를 YAML 파일에 저장합니다.

```bash
python -m baseline.transfer.transfer_pretrain
```

### 4. Non-transfer 실행

Non-transfer 학습 및 평가는 `baseline/non_transfer/non_transfer_learning.py`에서 진행합니다:

```bash
python -m baseline.non_transfer.non_transfer_learning
```

### 5. 오디오 시각화 (Streamlit 앱)

로컬 오디오 파일을 시각화하려면,  
`baseline/utils/visualization.py` 모듈을 기반으로 작성된 Streamlit 앱을 실행합니다.  
예시로 `streamlit_app.py`를 프로젝트 루트에 작성한 경우:

```python
import os
import streamlit as st
from baseline.utils.visualization import streamlit_audio_visualization

def main():
    st.title("오디오 시각화")
    st.write("로컬 디렉토리에서 오디오 파일을 선택하여 원본 파형과 Mel Spectrogram을 비교합니다.")
    
    # 예: /home/sh/Sound_Dataset 내의 파일들을 불러오기
    directory = "/home/sh/Sound_Dataset"
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))
    
    if not files:
        st.error("지정한 디렉토리 내에 WAV 파일이 없습니다.")
        return
    
    # 파일명을 선택
    selected_file = st.selectbox("오디오 파일 선택", files)
    st.write("선택한 파일:", selected_file)
    
    # 선택한 파일을 시각화
    streamlit_audio_visualization(selected_file)

if __name__ == "__main__":
    main()
```

Streamlit 앱을 실행하려면 다음 명령어를 사용하세요:

```bash
streamlit run streamlit_app.py
```

## 주의사항

- YAML 파일에 설정된 경로(예: base_directory, pickle_directory 등)가 실제 디렉토리 구조와 일치하는지 확인하세요.
- 대용량 중간 산출물(예: pickle 파일 등)은 필요시 `.gitignore`에 추가하여 Git에 포함되지 않도록 관리하는 것이 좋습니다.
- Git history에서 대용량 파일을 제거할 때는 Git LFS나 BFG Repo-Cleaner를 사용하여 기록을 재작성해야 합니다.
