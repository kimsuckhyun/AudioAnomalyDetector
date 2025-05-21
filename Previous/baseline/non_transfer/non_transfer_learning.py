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
from baseline.utils import (
    setup_logger,
    save_pickle,
    load_pickle,
    file_to_vector_array,
    list_to_vector_array,
    dataset_generator,
    Autoencoder
)

def main():
    # config 폴더 내의 non_transfer_learning.yaml 파일을 읽어옵니다.
    with open(os.path.join("config", "non_transfer_learning.yaml"), encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger("non_transfer_learning_torch.log")
    os.makedirs(config["pickle_directory"], exist_ok=True)
    os.makedirs(config["model_directory"], exist_ok=True)
    os.makedirs(config["result_directory"], exist_ok=True)
    
    model_directory = config["model_directory"]
    base_directory = config["base_directory"]
    result_file = os.path.join(config["result_directory"], config["result_file"])
    results = {}
    
    # base_directory 아래의 모든 DB/기계/ID 폴더를 검색합니다.
    # 예: /home/sh/AudioAnomalyDetector/sample_data/DB1/machine1/id_00
    dirs = sorted(glob.glob(os.path.join(base_directory, "*", "*", "*")))
    if not dirs:
        logger.error("No target directories found under %s", base_directory)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print(f"[{dir_idx+1}/{len(dirs)}] {target_dir}")
        
        # 경로 예: /home/sh/AudioAnomalyDetector/sample_data/DB1/machine1/id_00
        db = os.path.basename(os.path.dirname(os.path.dirname(target_dir)))
        machine_type = os.path.basename(os.path.dirname(target_dir))
        machine_id = os.path.basename(target_dir)
        
        # 피클 파일 이름 설정 (재사용)
        train_pickle = os.path.join(config["pickle_directory"], f"train_{machine_type}_{machine_id}_{db}.pickle")
        eval_files_pickle = os.path.join(config["pickle_directory"], f"eval_files_{machine_type}_{machine_id}_{db}.pickle")
        eval_labels_pickle = os.path.join(config["pickle_directory"], f"eval_labels_{machine_type}_{machine_id}_{db}.pickle")
        
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle, logger)
            eval_files = load_pickle(eval_files_pickle, logger)
            eval_labels = load_pickle(eval_labels_pickle, logger)
        else:
            # dataset_generator 함수는 target_dir 내의 normal/ 및 abnormal/ 폴더에서 데이터를 수집합니다.
            train_files, train_labels, eval_files, eval_labels = dataset_generator(
                target_dir,
                normal_dir_name="normal",
                abnormal_dir_name="abnormal",
                ext="wav",
                logger=logger
            )
            train_data = list_to_vector_array(
                train_files,
                n_mels=config["feature"]["n_mels"],
                frames=config["feature"]["frames"],
                n_fft=config["feature"]["n_fft"],
                hop_length=config["feature"]["hop_length"],
                power=config["feature"]["power"],
                logger=logger
            )
            save_pickle(train_pickle, train_data, logger)
            save_pickle(eval_files_pickle, eval_files, logger)
            save_pickle(eval_labels_pickle, eval_labels, logger)
        
        logger.info("Training non-transfer model for %s, %s, %s", machine_type, machine_id, db)
        input_dim = config["feature"]["n_mels"] * config["feature"]["frames"]
        model = Autoencoder(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
        criterion = nn.MSELoss()
        
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        dataset_all = TensorDataset(train_tensor, train_tensor)
        val_split = config["fit"].get("validation_split", 0)
        if val_split > 0:
            n_val = int(len(dataset_all) * val_split)
            n_train = len(dataset_all) - n_val
            train_dataset, val_dataset = random_split(dataset_all, [n_train, n_val])
            train_loader = DataLoader(train_dataset, batch_size=config["fit"]["batch_size"], shuffle=config["fit"]["shuffle"])
            val_loader = DataLoader(val_dataset, batch_size=config["fit"]["batch_size"], shuffle=False)
        else:
            train_loader = DataLoader(dataset_all, batch_size=config["fit"]["batch_size"], shuffle=config["fit"]["shuffle"])
            val_loader = None
        
        epochs = config["fit"]["epochs"]
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_x)
                        val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(val_loader.dataset)
                print(f"Validation Loss: {val_loss:.6f}")
        
        model_path = os.path.join(model_directory, f"model_{machine_type}_{machine_id}_{db}.pt")
        torch.save(model.state_dict(), model_path)
        print("Model saved to:", model_path)
        logger.info("Model saved to: %s", model_path)
        
        # 평가: 각 eval 파일에 대해 재구성 오차(MSE)를 계산하여 anomaly score로 사용
        y_pred = []
        model.eval()
        for file in tqdm(eval_files, desc="Evaluating", leave=False):
            try:
                data = file_to_vector_array(
                    file,
                    n_mels=config["feature"]["n_mels"],
                    frames=config["feature"]["frames"],
                    n_fft=config["feature"]["n_fft"],
                    hop_length=config["feature"]["hop_length"],
                    power=config["feature"]["power"],
                    logger=logger
                )
                if data.shape[0] == 0:
                    continue
                data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = model(data_tensor)
                error = torch.mean((data_tensor - outputs)**2, dim=1).cpu().numpy()
                y_pred.append(np.mean(error))
            except Exception as e:
                logger.warning("File broken: %s, Error: %s", file, str(e))
                y_pred.append(0.0)
        
        try:
            auc_score = metrics.roc_auc_score(eval_labels, y_pred)
            logger.info("AUC for %s_%s_%s: %.6f", machine_type, machine_id, db, auc_score)
            key = f"{machine_type}_{machine_id}_{db}"
            results[key] = {"AUC": float(auc_score)}
        except Exception as e:
            logger.error("Error computing ROC AUC: %s", e)
    
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info("All results saved to: %s", result_file)

if __name__ == "__main__":
    main()
