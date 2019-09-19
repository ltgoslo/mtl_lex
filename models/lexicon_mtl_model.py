import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from Utils.sst import *
from Utils.WordVecs import *
from model_training import *

import argparse



class MTL_lexicon(nn.Module):

    def __init__(self, word2idx,
                 embedding_matrix,
                 sentiment_label_size,
                 lexicon_label_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers=2,
                 lstm_dropout=0.2,
                 word_dropout=0.5,
                 train_embeddings=False):
        super(MTL_lexicon, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = word2idx
        self.vocab_size = len(word2idx)
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.sentiment_criterion = nn.CrossEntropyLoss()



        weight = torch.FloatTensor(embedding_matrix)
        self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=False)
        self.word_embeds.requires_grad = train_embeddings

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            dropout=lstm_dropout,
                            bidirectional=True)

        # lexicon prediction
        self.l1 = nn.Linear(embedding_dim, embedding_dim)
        self.l2 = nn.Linear(embedding_dim, lexicon_label_size)

        # Set up layers for sentiment prediction
        self.word_dropout = nn.Dropout(word_dropout)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.linear = nn.Linear(hidden_dim*2, sentiment_label_size)

    def project(self, x):
        e = self.word_embeds(x)
        o = F.relu(self.l1(e))
        o = F.softmax(self.l2(o), dim=1)
        return o

    def projection_loss(self, x, y):
        y_h = self.project(x)
        return self.sentiment_criterion(y_h, y)


    def init_hidden(self, batch_size=1):
        h0 = torch.zeros((self.lstm.num_layers*(1+self.lstm.bidirectional),
                                  batch_size, self.lstm.hidden_size))
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def max_pool(self, x):
        batch_size = x.batch_sizes[0]

        emb = self.word_embeds(x.data)

        projected = self.l1(emb)

        normed = self.batch_norm(emb)
        dropped = self.word_dropout(emb)

        padded = PackedSequence(dropped, x.batch_sizes)
        self.hidden = self.init_hidden(batch_size)

        lstm_out, (hn, cn) = self.lstm(padded, self.hidden)
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        pooled, _ = unpacked.max(dim=1)

        out = self.linear(pooled)
        return out

    def predict_sentiment(self, x):
        scores = self.max_pool(x)
        probs = F.softmax(scores, dim=1)
        preds = probs.argmax(dim=1)
        return preds

    def pooled_sentiment_loss(self, sents, labels):
        pred = self.max_pool(sents)
        loss = self.sentiment_criterion(pred, labels.flatten())
        return loss

    def eval_aux(self, X, Y):
        out = self.project(X)
        preds = out.argmax(dim=1).numpy()
        acc = accuracy_score(Y.numpy(), preds)
        print("Aux Acc: {0:.3f}".format(acc))
        return acc

    def predict_sentence(self, sentence):
        tokens = sentence.split()
        x = torch.tensor(self.vocab.ws2ids(tokens))
        y_h = self.project(x)
        print(" ".join(["{0}/neg:{1:.1f}|pos:{2:.1f}".format(w, n, p) for w, (n, p) in zip(tokens, y_h)]))

    def eval_sent(self, dev, batch_size):
        preds = []
        xs = []
        ys = []

        with torch.no_grad():
            for sents, targets in DataLoader(dev, batch_size=batch_size,
                                             collate_fn=dev.collate_fn,
                                             shuffle=False):
                pred = self.predict_sentiment(sents)
                xs.extend(sents)
                for x, y in zip(pred, targets):
                    preds.append(int(x))
                    ys.append(int(y))
        f1 = f1_score(ys, preds, average="macro")
        acc = accuracy_score(ys, preds)
        print("Sentiment F1: {0:.3f}".format(f1))
        print("Sentiment Acc: {0:.3f}".format(acc))
        return f1, acc, preds, xs, ys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=300, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_true")
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--EMBEDDINGS", "-emb", default="../../embeddings/google.txt")

    args = parser.parse_args()
    print(args)


    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    embeddings = WordVecs(args.EMBEDDINGS)
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 1
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    sst = SSTDataset(vocab, False, "../data/sst-fine")

    maintask_train_iter = sst.get_split("train")
    maintask_dev_iter = sst.get_split("dev")
    maintask_test_iter = sst.get_split("test")

    maintask_loader = DataLoader(maintask_train_iter,
                                 batch_size=args.BATCH_SIZE,
                                 collate_fn=maintask_train_iter.collate_fn,
                                 shuffle=True)


    lexicon_dataset = LexiconDataset(vocab, False, "../data/huliu_lexicon")
    lex_train = lexicon_dataset.get_split("train")
    lex_test = lexicon_dataset.get_split("test")

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, 300))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))

    model = MTL_lexicon(vocab,
                       new_matrix,
                       5,
                       2,
                       300,
                       100,
                       num_layers=2,
                       lstm_dropout=0.2,
                       word_dropout=0.5,
                       train_embeddings=False)

    batch_size = 10
    num_batches = (len(lex_train) // batch_size) + 1
    optimizer = optim.Adam(model.parameters())


    print("Training only lexicon parameters")
    for epoch in range(10):
        print("epoch {0}".format(epoch + 1))
        i = 0

        full_loss = 0

        for j in range(num_batches):
            x, y = zip(*lex_train[i:i+batch_size])
            i += batch_size
            x = torch.tensor(x)
            y = torch.tensor(y)
            model.zero_grad()
            loss = model.projection_loss(x, y)
            full_loss += loss
            loss.backward()
            optimizer.step()

        print("Loss: {0:.3f}".format(full_loss / num_batches))
        print("Train acc")
        x, y = zip(*lex_train)
        x = torch.tensor(x)
        y = torch.tensor(y)
        acc = model.eval_aux(x, y)
        print()

        print("Eval Acc")
        x, y = zip(*lex_test)
        x = torch.tensor(x)
        y = torch.tensor(y)
        acc = model.eval_aux(x, y)
        print()
