import torch.nn as nn

# Architecture - 1
class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]  # Getting the last sequence output
        x = self.fc(x)
        return x

# Architecture - 2
class Transformer(nn.Module):
    def __init__(self, embedding_dim, lstm_units_1, dropout_1, lstm_units_2, dropout_2, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=2, num_encoder_layers=2, num_decoder_layers=2)
        self.lstm1 = nn.LSTM(embedding_dim, lstm_units_1, batch_first=True, dropout=dropout_1, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2, batch_first=True, dropout=dropout_2, bidirectional=True)
        self.fc = nn.Linear(lstm_units_2 * 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x_tgt = x.clone()
        x = self.transformer(x, x_tgt)
        x = x.permute(1, 0, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x
    
# Architecture - 3
class BERTLSTM(nn.Module):
    def __init__(self, bert_model, lstm_units_1, dropout_1, lstm_units_2, dropout_2, num_classes):
        super(BERTLSTM, self).__init__()
        self.bert = bert_model
        self.lstm1 = nn.LSTM(bert_model.config.hidden_size, lstm_units_1, batch_first=True, dropout=dropout_1, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2, batch_first=True, dropout=dropout_2, bidirectional=True)
        self.fc = nn.Linear(lstm_units_2 * 2, num_classes)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids).last_hidden_state #BERT embeddings
        cls_token = bert_output[:, 0, :]
        cls_token = cls_token.unsqueeze(1)

        # LSTM layers
        x, _ = self.lstm1(cls_token)
        x, _ = self.lstm2(x)

        x = self.fc(x[:, -1, :])
        return x