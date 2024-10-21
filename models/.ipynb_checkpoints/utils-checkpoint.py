import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

from .lstm import LSTMModel, LSTMModel2
from .gru import GRUModel
from .lstm_attention import LSTMWithAttention, BiLSTMWithAttention

def create_model(model_name, input_size, hidden_size, num_layers, output_size, dropout=0.2):
    if model_name == 'GRU':
        return GRUModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LSTM':
        return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LSTM_Attention':
        # return BiLSTMWithAttention(input_size, hidden_size, num_layers, output_size, dropout)
        return LSTMWithAttention(input_size, hidden_size, num_layers, output_size, dropout)
    else:
        raise ValueError(f"Model {model_name} is not recognized. Please use 'GRU' or 'LSTM'.")

def train_and_evaluate(model, model_name, train_sequences, train_targets, 
                       eval_sequences, eval_targets, model_path, num_epochs=100, 
                       batch_size=64, learning_rate=0.001, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Data loader
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_sequences, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
    eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_sequences, dtype=torch.float32), torch.tensor(eval_targets, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-5)

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm을 사용하여 학습 진행도 표시
        with tqdm(train_loader, unit="batch") as tepoch:
            for sequences_batch, targets_batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
                
                # Move batch to GPU
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
                
                # Forward pass
                if 'Attention' in model_name:
                    outputs, _ = model(sequences_batch) 
                else:
                    outputs = model(sequences_batch)
                loss = criterion(outputs.squeeze(), targets_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 현재 배치 손실을 누적
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # Epoch 단위로 평균 손실 계산
        train_loss = running_loss / len(train_loader)

        # Evaluation on the validation set
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in eval_loader:
                # Move batch to GPU
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)

                if 'Attention' in model_name:
                    val_outputs, _ = model(sequences_batch) 
                else:
                    val_outputs = model(sequences_batch)
                loss = criterion(val_outputs.squeeze(), targets_batch)
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.4f}')

        # Check if the validation loss improved
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping applied at epoch {epoch+1}")
            break

def evaluate_r2_score(model, eval_sequences, eval_targets, model_name, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # 평가 모드로 전환

    eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_sequences, dtype=torch.float32), 
                                                  torch.tensor(eval_targets, dtype=torch.float32))
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for sequences_batch, targets_batch in eval_loader:
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
            if 'Attention' in model_name:
                outputs, _ = model(sequences_batch)
            else:
                outputs = model(sequences_batch)

            # 필요에 따라 squeeze() 적용
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)

            # GPU에서 CPU로 이동 및 리스트에 저장
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets_batch.cpu().numpy())

    # 리스트를 numpy 배열로 변환
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # R² 스코어 계산
    r2 = r2_score(all_targets, all_outputs)
    print(f"R² Score: {r2:.4f}")

    return r2

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model