import string

def clean_captions(mapping):
    for key, captions_list in mapping.items():
        for i in range(len(captions_list)):
            caption = captions_list[i]
            caption = caption.lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            words = caption.split()
            words = [word for word in words if len(word) > 1]
            caption = ' '.join(words)
            captions_list[i] = 'startseq ' + caption + ' endseq'