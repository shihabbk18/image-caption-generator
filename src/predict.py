import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    return in_text