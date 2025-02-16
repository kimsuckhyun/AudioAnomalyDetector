import os
import glob
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from baseline.utils import setup_logger, save_pickle, load_pickle, list_to_vector_array, Autoencoder

def main():
    # YAML 설정 파일 로드 (예: config/make_pretrain_v1.yaml)
    with open(os.path.join("config", "make_pretrain_v1.yaml"), encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger("make_pretrain_v1_torch.log")
    os.makedirs(config["pickle_directory"], exist_ok=True)
    os.makedirs(config["model_directory"], exist_ok=True)
    
    base_dir = config["base_directory"]  # 예: /home/sh/AudioAnomalyDetector/sample_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # base_dir 하위의 모든 "*dB_*" 디렉토리를 찾습니다. (예: -6dB_fan, 0dB_fan, 6dB_pump 등)
    machine_dirs = sorted(glob.glob(os.path.join(base_dir, "*dB_*")))
    if not machine_dirs:
        logger.error("No machine directories found under %s", base_dir)
        return

    # 각 dB×장치 디렉토리마다
    for machine_dir in machine_dirs:
        machine_basename = os.path.basename(machine_dir)  # 예: -6dB_fan
        logger.info("Processing machine directory: %s", machine_basename)
        
        # 장치 디렉토리 내부에 실제 장치 폴더가 있다고 가정 (예: fan)
        device_dirs = sorted(glob.glob(os.path.join(machine_dir, "*")))
        if not device_dirs:
            logger.warning("No device subdirectories found in %s, skipping...", machine_dir)
            continue
        
        for dev_dir in device_dirs:
            device_name = os.path.basename(dev_dir)  # 예: fan
            # id 디렉토리를 찾습니다. (예: id_00, id_02, id_06)
            id_dirs = sorted(glob.glob(os.path.join(dev_dir, "id_*")))
            if not id_dirs:
                logger.warning("No id directories found in %s, skipping...", dev_dir)
                continue
            
            for id_dir in id_dirs:
                id_name = os.path.basename(id_dir)  # 예: id_00
                logger.info("Processing target: %s %s %s", machine_basename, device_name, id_name)
                # normal 폴더 내의 WAV 파일들만 수집 (학습 데이터로 사용)
                normal_dir = os.path.join(id_dir, "normal")
                wav_files = sorted(glob.glob(os.path.join(normal_dir, "*.wav")))
                logger.info("Found %d WAV files in %s", len(wav_files), normal_dir)
                if not wav_files:
                    logger.warning("No WAV files in %s, skipping...", normal_dir)
                    continue
                
                # 피클 파일 이름: pretrain_only_<machine_basename>_<id_name>.pickle
                pickle_file = os.path.join(config["pickle_directory"],
                                           f"pretrain_only_{machine_basename}_{id_name}.pickle")
                if os.path.exists(pickle_file):
                    train_data = load_pickle(pickle_file)
                else:
                    from baseline.utils.feature import list_to_vector_array
                    train_data = list_to_vector_array(
                        wav_files,
                        n_mels=config["feature"]["n_mels"],
                        frames=config["feature"]["frames"],
                        n_fft=config["feature"]["n_fft"],
                        hop_length=config["feature"]["hop_length"],
                        power=config["feature"]["power"]
                    )
                    save_pickle(pickle_file, train_data)
                
                logger.info("Starting training for %s %s %s", machine_basename, device_name, id_name)
                input_dim = config["feature"]["n_mels"] * config["feature"]["frames"]
                model = Autoencoder(input_dim).to(device)
                optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
                criterion = nn.MSELoss()
                
                train_tensor = torch.tensor(train_data, dtype=torch.float32)
                dataset_ = TensorDataset(train_tensor, train_tensor)
                batch_size = config["fit"]["batch_size"]
                train_loader = DataLoader(dataset_, batch_size=batch_size, shuffle=config["fit"]["shuffle"])
                epochs = config["fit"]["epochs"]
                
                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0.0
                    for batch, _ in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch)
                        loss = criterion(outputs, batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * batch.size(0)
                    epoch_loss /= len(train_loader.dataset)
                    logger.info("Model %s_%s - Epoch [%d/%d] Loss: %.6f", device_name, id_name, epoch+1, epochs, epoch_loss)
                
                # 모델 저장: pretrain_only_<machine_basename>_<id_name>.pt
                model_file = os.path.join(config["model_directory"],
                                          f"pretrain_only_{machine_basename}_{id_name}.pt")
                torch.save(model.state_dict(), model_file)
                logger.info("Model saved to: %s", model_file)

if __name__ == "__main__":
    main()
