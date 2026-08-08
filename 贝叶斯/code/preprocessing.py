import re
import string

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label = parts[0]
                text = '\t'.join(parts[1:])
                X.append(preprocess(text))
                y.append(label)
    return X, y