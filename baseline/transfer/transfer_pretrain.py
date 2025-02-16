import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn import metrics
from tqdm import tqdm
from aad.utils import setup_logger, save_pickle, load_pickle, list_to_vector_array, dataset_generator, file_to_vector_array, Autoencoder

def main():
    with open(os.path.join("config", "make_pretrain_v1.yaml"), encoding='utf-8') as f:
        pretrain_config = yaml.safe_load(f)
    with open(os.path.join("config", "transfer_pretrain_v1.yaml"), encoding='utf-8') as f:
        transfer_config = yaml.safe_load(f)
    
    logger = setup_logger("transfer_pretrain_v1_torch.log")
    os.makedirs(transfer_config["pickle_directory"], exist_ok=True)
    os.makedirs(transfer_config["model_directory"], exist_ok=True)
    
    base_dir = transfer_config["base_directory"]
    machine_types = ["fan", "valve", "slider", "pump"]
    id_list = ["id_00", "id_02", "id_04", "id_06"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for machine_type in machine_types:
        for id_ in id_list:
            logger.info("Processing machine_type: %s, id: %s", machine_type, id_)
            train_pickle = os.path.join(transfer_config["pickle_directory"], f"train_{machine_type}_{id_}.pickle")
            eval_files_pickle = os.path.join(transfer_config["pickle_directory"], f"eval_files_{machine_type}_{id_}.pickle")
            eval_labels_pickle = os.path.join(transfer_config["pickle_directory"], f"eval_labels_{machine_type}_{id_}.pickle")
            
            if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
                train_data = load_pickle(train_pickle, logger)
                eval_files = load_pickle(eval_files_pickle, logger)
                eval_labels = load_pickle(eval_labels_pickle, logger)
            else:
                target_dir = os.path.join(base_dir, machine_type, id_)
                train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir,
                                                                                      normal_dir_name="normal",
                                                                                      abnormal_dir_name="abnormal",
                                                                                      ext="wav",
                                                                                      logger=logger)
                train_data = list_to_vector_array(train_files,
                                                  n_mels=transfer_config["feature"]["n_mels"],
                                                  frames=transfer_config["feature"]["frames"],
                                                  n_fft=transfer_config["feature"]["n_fft"],
                                                  hop_length=transfer_config["feature"]["hop_length"],
                                                  power=transfer_config["feature"]["power"],
                                                  logger=logger)
                save_pickle(train_pickle, train_data, logger)
                save_pickle(eval_files_pickle, eval_files, logger)
                save_pickle(eval_labels_pickle, eval_labels, logger)
            
            for source_type in machine_types:
                if source_type == machine_type:
                    continue
                logger.info("Transfer learning from %s to %s_%s", source_type, machine_type, id_)
                pretrain_model_file = os.path.join(pretrain_config["model_directory"], f"pretrain_only_-6dB_{source_type}.pt")
                input_dim = transfer_config["feature"]["n_mels"] * transfer_config["feature"]["frames"]
                model = Autoencoder(input_dim).to(device)
                model.load_state_dict(torch.load(pretrain_model_file, map_location=device))
                
                optimizer = optim.Adam(model.parameters(), lr=transfer_config.get("learning_rate", 0.001))
                criterion = nn.MSELoss()
                
                train_tensor = torch.tensor(train_data, dtype=torch.float32)
                dataset_ = TensorDataset(train_tensor, train_tensor)
                train_loader = DataLoader(dataset_, batch_size=transfer_config["fit"]["batch_size"], shuffle=transfer_config["fit"]["shuffle"])
                epochs = transfer_config["fit"]["epochs"]
                
                model.train()
                for epoch in range(epochs):
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
                    logger.info("Transfer Epoch [%d/%d] for %s-%s->%s: Loss %.6f", epoch+1, epochs, machine_type, id_, source_type, total_loss)
                
                transfer_model_file = os.path.join(transfer_config["model_directory"], f"transfer_{machine_type}_{id_}_to_{source_type}.pt")
                torch.save(model.state_dict(), transfer_model_file)
                logger.info("Transfer model saved to: %s", transfer_model_file)
                
                y_pred = []
                model.eval()
                for file in tqdm(eval_files, desc="Evaluating", leave=False):
                    data = file_to_vector_array(file,
                                                n_mels=transfer_config["feature"]["n_mels"],
                                                frames=transfer_config["feature"]["frames"],
                                                n_fft=transfer_config["feature"]["n_fft"],
                                                hop_length=transfer_config["feature"]["hop_length"],
                                                power=transfer_config["feature"]["power"],
                                                logger=logger)
                    if data.shape[0] == 0:
                        continue
                    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        outputs = model(data_tensor)
                    error = torch.mean((data_tensor - outputs)**2, dim=1).cpu().numpy()
                    file_error = error.mean()
                    y_pred.append(file_error)
                
                try:
                    score = metrics.roc_auc_score(eval_labels, y_pred)
                    logger.info("ROC AUC for %s_%s to %s: %.6f", machine_type, id_, source_type, score)
                    results[f"{machine_type}_{id_}_to_{source_type}"] = {"AUC": float(score)}
                except Exception as e:
                    logger.error("Error computing ROC AUC: %s", e)
    
    result_file = os.path.join(transfer_config["result_directory"], transfer_config["result_file"])
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info("All results saved to: %s", result_file)

if __name__ == "__main__":
    main()