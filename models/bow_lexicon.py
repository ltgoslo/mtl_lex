import sys
import argparse
import numpy as np
from Utils.sst import *
from Utils.WordVecs import *
from model_training import *


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

def count(sent_idxs, sent_lex):
    c = 0
    for i in sent_idxs:
        if int(i) in sent_lex:
            c += 1
    return c

def bow_lexicon(sent_idxs, vocab, pos, neg):
    # Create bag of word representations for the sentence.
    s = np.zeros((len(vocab)))

    for i, idx in enumerate(sent_idxs):
        s[idx] += 1
    c_pos = [count(sent_idxs, pos)]
    c_neg = [count(sent_idxs, neg)]
    return np.concatenate((s, c_pos, c_neg))


def get_features(X, w2idx, pos, neg):
    return [bow_lexicon(s, w2idx, pos, neg) for s in X]

def get_c(trainX, trainy,
          devX, devy):

    best_c = 0
    best_f1 = 0

    for c in [0.0001, 0.0003, 0.001, 0.003,
              0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        clf = LinearSVC(C=c)
        h = clf.fit(trainX, trainy)
        pred = clf.predict(devX)
        dev_f1 = f1_score(devy, pred, average="macro")
        if dev_f1 > best_f1:
            best_c = c
            best_f1 = dev_f1

    print("best f1: {0:.3f} with c={1}".format(best_f1, best_c))
    return best_f1, best_c

def main():
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATASET", "-data",
                        default="sst")
    parser.add_argument("--LEXICON", "-lex",
                        default="huliu_lexicon")

    args = parser.parse_args()
    print(args)

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Import datasets

    neg = [l.strip() for l in open("../data/{0}/negative-words.txt".format(args.LEXICON), encoding="latin1")]
    pos = [l.strip() for l in open("../data/{0}/positive-words.txt".format(args.LEXICON), encoding="latin1")]

    neg = vocab.ws2ids(neg)
    pos = vocab.ws2ids(pos)


    # This will update vocab with words not found in embeddings
    sst = SSTDataset(vocab, False, "../data/{0}-fine".format(args.DATASET))

    maintask_train_iter = sst.get_split("train")
    maintask_dev_iter = sst.get_split("dev")
    maintask_test_iter = sst.get_split("test")


    X, Y = zip(*maintask_train_iter)
    trainX = get_features(X, vocab, pos, neg)
    trainy = Y

    X, Y = zip(*maintask_dev_iter)
    devX = get_features(X, vocab, pos, neg)
    devy = Y

    X, Y = zip(*maintask_test_iter)
    testX = get_features(X, vocab, pos, neg)
    testy = Y

    best_f1, best_c = get_c(trainX, trainy,
                            devX, devy)
    #best_c = 0.1

    clf = LinearSVC(C=best_c)
    h = clf.fit(trainX, trainy)
    pred = clf.predict(testX)

    print("Macro F1: {0:.3f}".format(f1_score(testy, pred, average="macro")))
