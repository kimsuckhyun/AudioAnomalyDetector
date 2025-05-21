# AudioAnomalyDetector/src/utils/seed.py

import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """
    랜덤 시드를 고정하여 재현성을 확보합니다.

    Args:
        seed (int): 설정할 시드 값 (기본: 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # CUDA 사용 시에도 동일한 시드를 고정하려면 아래 두 줄을 추가할 수 있습니다.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)