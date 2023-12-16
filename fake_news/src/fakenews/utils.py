import torch

from fakenews.preprocess import clean_text
from keras.preprocessing.sequence import pad_sequences

def predict_text(text, model, tokenizer, max_length, device):
    text = clean_text(text)
    input_sequence = tokenizer.texts_to_sequences([text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    input_tensor = torch.tensor(input_sequence, dtype=torch.long)

    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(input_tensor), dim=1)
    predicted_class = torch.argmax(output).item()

    class_mapping = {0: "fake", 1: "real"}
    predicted_label = class_mapping[predicted_class]

    return predicted_label