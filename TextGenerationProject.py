import numpy
import sys
import nltk
import tensorflow as tf
from tensorflow import keras
nltk.download("stopwords")
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM 
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 
  
  
file = open("frankenstein-2.txt").read()

def tokenize_words(input):
    input = input.lower()
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(input)
    filtered = filter( lambda token : token not in stopwords.words("english"),tokens)
    return "  ".join(filtered)
processed_inputs = tokenize_words(file)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i,c in enumerate(chars))



input_len = len(processed_inputs)
vocab_len = len(chars)
print("The total number of characters:", input_len)
print("Total vocab:" , vocab_len)

for i in range(0,input_len - seq_length,1):
    in_seq = processed_inputs(i:i + seq_length)
    out_seq = processed_inputs(i + seq_length)
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
print("Total Pattern:",n_patterns)


X = numpy.reshape(x_data, (n_patterns , seq_length , 1))
X = X/float(vocab_len) 



Y = np_utils.to_categorical(y_data)


models = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(256, return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation = "softmax"))

filename = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor="loss",verbose = 1,save_best_only=True, mode = "min)











































