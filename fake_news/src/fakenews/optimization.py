import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertModel
from fakenews.model import LSTM, Transformer, BERTLSTM

def lstm_objective(trial, train_loader, test_loader, device):
    # Hyperparameters
    embedding_dim = trial.suggest_int('embedding_dim', 64, 256, step=32)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    input_dim = 5000
    output_dim = 2

    # Model, loss, and optimizer
    model = LSTM(input_dim, embedding_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch.long())
            loss = criterion(output, y_batch.float())
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch.long())

            predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
            true_class = y_batch.argmax(1)
            correct += (predicted == true_class).sum().item()
            total += y_batch.size(0)

    val_accuracy = correct / total
    return val_accuracy

def transformer_objective(trial, train_loader, test_loader, device):
    # Hyperparameters
    embedding_dim = trial.suggest_int('embedding_dim', 64, 256, step=32)
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=32)
    dropout_1 = trial.suggest_float('dropout_1', 0.2, 0.5, step=0.1)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=32)
    dropout_2 = trial.suggest_float('dropout_2', 0.2, 0.5, step=0.1)

    vocab_size = 5000

    # Model, loss, and optimizer
    model = Transformer(embedding_dim, lstm_units_1, dropout_1, lstm_units_2, dropout_2, vocab_size)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).long(), y_batch.to(device).long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device).long(), y_batch.to(device).long()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    return val_loss / len(test_loader)

def bertlstm_objective(trial, train_loader, test_loader, device):
    # Hyperparameters
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=32)
    dropout_1 = trial.suggest_float('dropout_1', 0.2, 0.5, step=0.1)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=32)
    dropout_2 = trial.suggest_float('dropout_2', 0.2, 0.5, step=0.1)

    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Model, loss, and optimizer
    model = BERTLSTM(bert_model, lstm_units_1, dropout_1, lstm_units_2, dropout_2, 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).long(), y_batch.to(device).long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device).long(), y_batch.to(device).long()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    return val_loss / len(test_loader)