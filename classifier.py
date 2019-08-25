import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


list_sentences_train = np.array([1])
max_features, max_len = 20000, 100
tokenizer = Tokenizer(num_words=max_features)  # 20000 i.e. num_words to use


def warm_up():
    data_dir = 'resources/'
    global list_sentences_train
    global tokenizer
    train = pd.read_csv(os.path.join(data_dir, 'train.zip'))
    list_sentences_train = train['comment_text'].fillna('_na_').values


def predict(text):
    K.clear_session()
    data_dir = 'resources/'

    # train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    # list_sentences_train = train['comment_text'].fillna('_na_').values

    text = np.array(text).reshape(1, )

    # tokenizer = Tokenizer(num_words=max_features)  # 20000 i.e. num_words to use
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(list_tokenized_test, maxlen=max_len)  # Restrict each row in df to max_len = 100 words

    with open(os.path.join(data_dir, 'model.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    model.load_weights(os.path.join(data_dir, 'weights.h5'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_test = model.predict([x_test], verbose=1)

    if len(np.nonzero(y_test > 0.5)[0]) == 0:
        predictions = ['Non-Toxic']
    else:
        list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        predictions = [list_classes[idx] for idx in np.nonzero(y_test > 0.5)[1]]

    K.clear_session()
    return predictions
