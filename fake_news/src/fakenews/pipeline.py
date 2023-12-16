import torch
import pickle

from fakenews.utils import predict_text
from fakenews.model import LSTM

def pipeline(text):
    model_path = 'models/' + 'lstm.pth'
    tokenizer = 'tokenizer/' + 'lstm.pickle'
    max_seq_len = 228

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path)
    loaded_model = LSTM(
        input_dim=checkpoint['input_dim'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    loaded_model.to(device)
    loaded_model.eval()

    # Loading the saved tokenizer
    with open(tokenizer, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    predicted_label = predict_text(text, loaded_model, loaded_tokenizer, max_seq_len, device)
    return predicted_label