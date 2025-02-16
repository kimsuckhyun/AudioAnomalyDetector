import os
import glob
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from aad.utils import setup_logger, save_pickle, load_pickle, list_to_vector_array, Autoencoder
# dataset_generator는 여기서 사용할 필요에 따라 추가

def main():
    with open(os.path.join("config", "make_pretrain_v1.yaml"), encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger("make_pretrain_v1_torch.log")
    
    os.makedirs(config["pickle_directory"], exist_ok=True)
    os.makedirs(config["model_directory"], exist_ok=True)
    
    base_dir = config["base_directory"]
    machine_types = ["fan", "valve", "slider", "pump"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for machine_type in machine_types:
        db = f"-6dB_{machine_type}"
        train_pickle = os.path.join(config["pickle_directory"], f"pretrain_only_{db}.pickle")
        
        if os.path.exists(train_pickle):
            train_data = load_pickle(train_pickle, logger)
        else:
            # 예시: target_dir 내 normal 폴더의 모든 .wav 파일을 사용
            target_dir = os.path.join(base_dir, f"-6dB_{machine_type}", "normal")
            train_files = sorted(glob.glob(os.path.join(target_dir, "*.wav")))
            from aad.utils.feature import list_to_vector_array
            train_data = list_to_vector_array(train_files,
                                              n_mels=config["feature"]["n_mels"],
                                              frames=config["feature"]["frames"],
                                              n_fft=config["feature"]["n_fft"],
                                              hop_length=config["feature"]["hop_length"],
                                              power=config["feature"]["power"],
                                              logger=logger)
            save_pickle(train_pickle, train_data, logger)
        
        logger.info("Starting pretrain for machine type: %s", machine_type)
        input_dim = config["feature"]["n_mels"] * config["feature"]["frames"]
        model = Autoencoder(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
        criterion = nn.MSELoss()
        
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        dataset_ = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(dataset_, batch_size=config["fit"]["batch_size"], shuffle=config["fit"]["shuffle"])
        epochs = config["fit"]["epochs"]
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch, _ in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)
            total_loss /= len(train_loader.dataset)
            logger.info("Epoch [%d/%d] Loss: %.6f", epoch+1, epochs, total_loss)
        
        model_file = os.path.join(config["model_directory"], f"pretrain_only_{db}.pt")
        torch.save(model.state_dict(), model_file)
        logger.info("Pretrain model saved to: %s", model_file)

if __name__ == "__main__":
    main()