import os
import glob
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import yaml

from baseline.utils import (
    setup_logger,
    save_pickle,
    load_pickle,
    list_to_vector_array,
    dataset_generator,
    file_to_vector_array,
    Autoencoder
)

def main():
    # YAML 설정 파일 로드
    with open(os.path.join("config", "make_pretrain_v1.yaml"), encoding='utf-8') as f:
        pretrain_config = yaml.safe_load(f)
    with open(os.path.join("config", "transfer_pretrain_v1.yaml"), encoding='utf-8') as f:
        transfer_config = yaml.safe_load(f)
    
    logger = setup_logger("transfer_pretrain_v1_torch.log")
    os.makedirs(transfer_config["pickle_directory"], exist_ok=True)
    os.makedirs(transfer_config["model_directory"], exist_ok=True)
    os.makedirs(transfer_config["result_directory"], exist_ok=True)
    
    base_dir = transfer_config["base_directory"]  # 예: /home/sh/AudioAnomalyDetector/sample_data
    # 대상 디렉토리: base_dir 하위의 모든 DB/장치/ID 폴더 (예: -6dB_fan/fan/id_00)
    target_dirs = sorted(glob.glob(os.path.join(base_dir, "*", "*", "*")))
    if not target_dirs:
        logger.error("No target directories found under %s", base_dir)
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transfer_config에 대상 장치 목록 지정 (예: fan, valve, slider, pump)
    device_list = transfer_config.get("device_list", ["fan", "valve", "slider", "pump"])
    
    results = {}
    
    for target_dir in target_dirs:
        # 예: target_dir = /home/sh/AudioAnomalyDetector/sample_data/-6dB_fan/fan/id_00
        # 부모 두 개 폴더에서 dB 및 장치 정보 추출
        dB_device = os.path.basename(os.path.dirname(os.path.dirname(target_dir)))  # 예: -6dB_fan
        parts = dB_device.split("_")
        if len(parts) < 2:
            logger.error("Cannot parse dB and device from %s", dB_device)
            continue
        dB_target = parts[0]      # 예: -6dB, 0dB, 6dB
        target_device = parts[1]  # 예: fan, valve, ...
        target_id = os.path.basename(target_dir)  # 예: id_00
        logger.info("Processing target: %s, %s, %s", dB_target, target_device, target_id)
        
        # 피클 파일명 설정 (대상 데이터)
        train_pickle = os.path.join(transfer_config["pickle_directory"], f"train_{target_device}_{target_id}_{dB_target}.pickle")
        eval_files_pickle = os.path.join(transfer_config["pickle_directory"], f"eval_files_{target_device}_{target_id}_{dB_target}.pickle")
        eval_labels_pickle = os.path.join(transfer_config["pickle_directory"], f"eval_labels_{target_device}_{target_id}_{dB_target}.pickle")
        
        # 대상 데이터가 이미 저장되어 있으면 불러오기, 없으면 생성
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle, logger)
            eval_files = load_pickle(eval_files_pickle, logger)
            eval_labels = load_pickle(eval_labels_pickle, logger)
        else:
            # dataset_generator는 대상 디렉토리 내의 normal/ 및 abnormal/ 폴더에서 데이터를 수집합니다.
            train_files, train_labels, eval_files, eval_labels = dataset_generator(
                target_dir,
                normal_dir_name="normal",
                abnormal_dir_name="abnormal",
                ext="wav",
                logger=logger
            )
            train_data = list_to_vector_array(
                train_files,
                n_mels=transfer_config["feature"]["n_mels"],
                frames=transfer_config["feature"]["frames"],
                n_fft=transfer_config["feature"]["n_fft"],
                hop_length=transfer_config["feature"]["hop_length"],
                power=transfer_config["feature"]["power"],
                logger=logger
            )
            save_pickle(train_pickle, train_data, logger)
            save_pickle(eval_files_pickle, eval_files, logger)
            save_pickle(eval_labels_pickle, eval_labels, logger)
        
        # 대상과 동일한 장치는 건너뛰고, 다른 장치의 pretrain 모델들을 이용하여 transfer 학습 수행
        for source in device_list:
            if source == target_device:
                continue
            # pretrain 모델 파일 패턴: pretrain_<dB>_<source>.pt (예: pretrain_-6dB_fan.pt)
            pattern = os.path.join(pretrain_config["model_directory"], f"pretrain_*_{source}.pt")
            source_model_files = sorted(glob.glob(pattern))
            if not source_model_files:
                logger.warning("No pretrain model files found for source device %s", source)
                continue
            
            for pretrain_model_file in source_model_files:
                basename = os.path.basename(pretrain_model_file)
                parts = basename.split("_")
                if len(parts) < 2:
                    logger.error("Cannot parse source dB from %s", basename)
                    continue
                source_dB = parts[1]  # 예: -6dB, 0dB, 6dB
                logger.info("Transfer learning from %s (%s) to %s %s (%s)", source, source_dB, target_device, target_id, dB_target)
                
                input_dim = transfer_config["feature"]["n_mels"] * transfer_config["feature"]["frames"]
                model_transfer = Autoencoder(input_dim).to(device)
                model_transfer.load_state_dict(torch.load(pretrain_model_file, map_location=device))
                
                optimizer = optim.Adam(model_transfer.parameters(), lr=transfer_config.get("learning_rate", 0.001))
                criterion = nn.MSELoss()
                
                train_tensor = torch.tensor(train_data, dtype=torch.float32)
                dataset_ = TensorDataset(train_tensor, train_tensor)
                train_loader = DataLoader(dataset_, batch_size=transfer_config["fit"]["batch_size"], shuffle=transfer_config["fit"]["shuffle"])
                epochs = transfer_config["fit"]["epochs"]
                
                model_transfer.train()
                for epoch in range(epochs):
                    total_loss = 0.0
                    for batch, _ in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        outputs = model_transfer(batch)
                        loss = criterion(outputs, batch)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * batch.size(0)
                    total_loss /= len(train_loader.dataset)
                    logger.info("Transfer Epoch [%d/%d] for %s-%s(%s) -> %s (%s): Loss %.6f", 
                                epoch+1, epochs, target_device, target_id, dB_target, source, source_dB, total_loss)
                
                transfer_model_file = os.path.join(
                    transfer_config["model_directory"],
                    f"transfer_{dB_target}_{target_device}_{target_id}_to_{source}_{source_dB}.pt"
                )
                torch.save(model_transfer.state_dict(), transfer_model_file)
                logger.info("Transfer model saved to: %s", transfer_model_file)
                
                # 평가: 대상 eval 파일에 대해 재구성 오차(MSE)를 anomaly score로 계산
                y_pred = []
                model_transfer.eval()
                for file in tqdm(eval_files, desc="Evaluating", leave=False):
                    try:
                        data = file_to_vector_array(
                            file,
                            n_mels=transfer_config["feature"]["n_mels"],
                            frames=transfer_config["feature"]["frames"],
                            n_fft=transfer_config["feature"]["n_fft"],
                            hop_length=transfer_config["feature"]["hop_length"],
                            power=transfer_config["feature"]["power"],
                            logger=logger
                        )
                        if data.shape[0] == 0:
                            continue
                        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            outputs = model_transfer(data_tensor)
                        error = torch.mean((data_tensor - outputs)**2, dim=1).cpu().numpy()
                        y_pred.append(np.mean(error))
                    except Exception as e:
                        logger.warning("File broken: %s, Error: %s", file, str(e))
                        y_pred.append(0.0)
                
                try:
                    score = metrics.roc_auc_score(eval_labels, y_pred)
                    logger.info("ROC AUC for %s-%s(%s) -> %s (%s): %.6f", target_device, target_id, dB_target, source, source_dB, score)
                    key = f"{target_device}_{target_id}_{dB_target}_to_{source}_{source_dB}"
                    results[key] = {"AUC": float(score)}
                except Exception as e:
                    logger.error("Error computing ROC AUC: %s", e)
    
    result_file = os.path.join(transfer_config["result_directory"], transfer_config["result_file"])
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info("All results saved to: %s", result_file)

if __name__ == "__main__":
    main()
