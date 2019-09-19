import numpy as np

def ave_vecs(sentence, model):
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += model['the']
    return sent / sent_length


def idx_vecs(sentence, model):
    """Returns a list of vectors of the tokens
    in the sentence if they are in the model."""
    sent = []
    for w in sentence.split():
        try:
            sent.append(model[w])
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent.append(model['of'])
    return sent


def words(sentence, model):
    return sentence


def getData(fname, model, representation=words, encoding='utf8'):
    X = []
    y = []
    for line in open(fname):
        label, text = line.split()[0], line.split()[1:]
        X.append(representation(text, model))
        y.append(int(label))
    return X, y

