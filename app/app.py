from flask import Flask, request, render_template
import torch
import torch.nn as nn
import tensorflow as tf
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

def clean_text(text):
    text = text.lower()#converts the text to lowercase to ensure consistency.
    text = re.sub(r'[^\w\s]', '', text)#Removes any characters that are not alphanumeric or whitespace.
    text = re.sub(r'\d+', " ", text)#Removes any digits by replacing them with a space.
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)#Removes single characters surrounded by whitespace.
    ###Expand the contractions###
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    ######################################
    text = re.sub(r"\'s", " ", text)#Removes apostrophes followed by an "s"
    text = re.sub('\W', ' ', text)#Removes any non-word characters by replacing them with a space.
    text = re.sub('\s+', ' ', text)#Replaces multiple consecutive spaces with a single space.
    text=text.translate(str.maketrans('', '', string.punctuation)) #Removes punctuation marks
    ###Removes stop words###
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    ##################################
    text=lemmatizing(text)
    text = text.strip(' ')
    return text

# Load the saved deep learning model
proj_dir = ''  # Update with latest directory path

loaded_model.load_state_dict(torch.load('lstm.pth'))
#loaded_model = tf.keras.models.load_model(proj_dir + "best_model.h5")

# Load the Tokenizer from the saved file
with open(proj_dir + 'tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        input_text = request.form['input_text']
        result = predict(input_text)
    return render_template('index.html', result=result)

def predict(input_text):
    text_to_tokenize = clean_text(input_text)
    tokenized_text = loaded_tokenizer.texts_to_sequences([text_to_tokenize])
    tokenized_text = pad_sequences(tokenized_text, maxlen=225)
    predictions = loaded_model.predict(tokenized_text)
    if np.argmax(predictions) == 0:
        return 'Fake'
    else:
        return 'Real'

if __name__ == '__main__':
    app.run(debug=True)
