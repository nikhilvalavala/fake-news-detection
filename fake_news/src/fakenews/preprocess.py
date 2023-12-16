import os
import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Labels to Binary Values
def Liar_labling(label):
    if label in ['half-true', 'mostly-true', 'true', 'barely-true']:
        label = 1
    elif label == 'false':
        label = 0
    return label

# Processing Liar Dataset
def Liar_data_set(loc):
    df = pd.read_csv(loc, delimiter='\t', header=None)
    df[1] = df[1].apply(lambda x: Liar_labling(x))
    df = df[df[1] != 'pants-fire'][[2, 1]] # Removing pants-fire label
    new_column_names = {1: 'label', 2: 'text'}
    df.rename(columns=new_column_names, inplace=True)
    return df

# Processing FakeNewsNet Dataset
def FakeNewsNet_data_set(loc):
    df = pd.read_csv(loc)
    file_name = loc.split('/')[-1]
    label = None

    # Labels
    if file_name.split('_')[1].split('.')[0] == 'fake':
        label = 0
    elif file_name.split('_')[1].split('.')[0] == 'real':
        label = 1

    df = df[['title']].copy()
    df.rename(columns={'title': 'text'}, inplace=True)
    if label is not None:
        df['label'] = label
    if label == 1:
        df = df[['text', 'label']].copy()
    return df

# Combine Liar and FakeNewsNet to one DataFrame
def combine_dataset(proj_dir):
    
    # Liar Dataset
    file_names = os.listdir(proj_dir + '/data/Liar')
    liar = pd.DataFrame() # Liar DataFrame

    for file_name in file_names:
        df = Liar_data_set(proj_dir + '/data/Liar/' + file_name)
        liar = pd.concat([liar, df])

    liar.reset_index(drop=True, inplace=True)
    # print("Liar Dataset\n", liar)
    liar.to_csv(proj_dir+'/data/liar.csv')


    # FakeNewsNet
    file_names = os.listdir(proj_dir + '/data/FakeNewsNet/')
    fkn = pd.DataFrame() # FakeNewsNet DataFrame

    for file_name in file_names:
        df = FakeNewsNet_data_set(proj_dir + '/data/FakeNewsNet/' + file_name)
        fkn = pd.concat([fkn, df])

    fkn.reset_index(drop=True, inplace=True)
    # print("\n\nFakeNewsNet Dataset\n", fkn)
    fkn.to_csv(proj_dir+'/data/fkn.csv')


    # Concatenation Liar and FakeNewsNet
    df = pd.concat([liar, fkn])
    df.reset_index(drop=True, inplace=True)
    df.to_csv(proj_dir+'/data/lstm_data.csv')

    return liar, fkn, df

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

# Cleaning text by applying various preprocessing steps
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', " ", text)
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text=text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text=lemmatizing(text)
    text = text.strip(' ')
    return text