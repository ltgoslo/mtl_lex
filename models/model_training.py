
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from Utils.WordVecs import WordVecs
from Utils.utils import *
import numpy as np


import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import defaultdict
from Utils.sst import SSTDataset
from torch.utils.data import DataLoader

import os
import argparse
import pickle

from lexicon_mtl_model import *


class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def train_model(vocab,
                new_matrix,
                num_labels,
                embedding_dim,
                hidden_dim,
                num_lstm_layers,
                train_embeddings,
                lex_train,
                lex_test,
                maintask_loader,
                maintask_train_iter,
                maintask_dev_iter,
                AUXILIARY_TASK=None,
                epochs=10,
                sentiment_learning_rate=0.001,
                auxiliary_learning_rate=0.0001,
                BATCH_SIZE=50,
                number_of_runs=5,
                random_seeds=[123, 456, 789, 101112, 131415]
                ):

    # Save the model parameters
    param_file = (dict(vocab.items()),
                  new_matrix.shape,
                  None,
                  num_labels,
                  None)

    basedir = os.path.join("saved_models",
                           args.DATASET,
                           args.FINE_GRAINED,
                           args.AUXILIARY_TASK,
                           args.LEXICON)
    outfile = os.path.join(basedir,
                           "params.pkl")
    os.makedirs(basedir, exist_ok=True)

    with open(outfile, "wb") as out:
        pickle.dump(param_file, out)

    for i, run in enumerate(range(number_of_runs)):

        model = MTL_lexicon(vocab,
                            new_matrix,
                            num_labels,
                            2,
                            embedding_dim,
                            hidden_dim,
                            num_layers=2,
                            lstm_dropout=0.2,
                            word_dropout=0.5,
                            train_embeddings=train_embeddings)

        # Set our optimizers
        sentiment_params = list(model.word_embeds.parameters()) + \
                           list(model.l1.parameters()) +\
                           list(model.lstm.parameters()) +\
                           list(model.linear.parameters())

        auxiliary_params = list(model.word_embeds.parameters()) + \
                           list(model.l1.parameters()) +\
                           list(model.l2.parameters())

        sentiment_optimizer = torch.optim.Adam(sentiment_params, lr=sentiment_learning_rate)
        auxiliary_optimizer = torch.optim.Adam(auxiliary_params, lr=auxiliary_learning_rate)

        print("RUN {0}".format(run + 1))
        best_dev_acc = 0.0

        # set random seed for reproducibility
        np.random.seed(random_seeds[i])
        torch.manual_seed(random_seeds[i])

        lex_batch_size = 50
        num_lex_batches = (len(lex_train) // lex_batch_size) + 1

        for j, epoch in enumerate(range(epochs)):

            # If AUXILIARY_TASK is None, defaults to single task
            if AUXILIARY_TASK not in ["None", "none", 0, None]:

                print("epoch {0}: ".format(epoch + 1), end="")
                print("Training auxiliary task...")

                full_loss = 0
                i = 0
                for batch in tqdm(range(num_lex_batches)):
                    x, y = zip(*lex_train[i:i+lex_batch_size])
                    i += lex_batch_size
                    x = torch.tensor(x)
                    y = torch.tensor(y)
                    model.zero_grad()
                    loss = model.projection_loss(x, y)
                    full_loss += loss
                    loss.backward()
                    auxiliary_optimizer.step()

                print("Loss: {0:.3f}".format(full_loss / num_lex_batches))
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


            batch_losses = 0
            num_batches = 0
            model.train()

            print("epoch {0}: Training main task...".format(epoch + 1))

            for sents, targets in maintask_loader:
                model.zero_grad()

                loss = model.pooled_sentiment_loss(sents, targets)
                batch_losses += loss.data
                num_batches += 1

                loss.backward()
                sentiment_optimizer.step()

            print()
            print("loss: {0:.3f}".format(batch_losses / num_batches))
            model.eval()
            f1, acc, preds, ys = model.eval_sent(maintask_train_iter,
                                                 batch_size=BATCH_SIZE)
            f1, acc, preds, ys = model.eval_sent(maintask_dev_iter,
                                                 batch_size=BATCH_SIZE)

            if acc > best_dev_acc:
                best_dev_acc = acc
                print("NEW BEST DEV ACC: {0:.3f}".format(acc))


                basedir = os.path.join("saved_models",
                                       args.DATASET,
                                       args.FINE_GRAINED,
                                       AUXILIARY_TASK,
                                       args.LEXICON,
                                       "{0}".format(run + 1))
                outname = "epochs:{0}-lstm_dim:{1}-lstm_layers:{2}-devacc:{3:.3f}".format(epoch + 1, model.lstm.hidden_size, model.lstm.num_layers, acc)
                modelfile = os.path.join(basedir,
                                         outname)
                os.makedirs(basedir, exist_ok=True)
                print("saving model to {0}".format(modelfile))
                torch.save(model.state_dict(), modelfile)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=300, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_false")
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="lexicon_prediction")
    parser.add_argument("--EMBEDDINGS", "-emb", default="../../embeddings/blse/google.txt")
    parser.add_argument("--SENTIMENT_LR", "-slr", default=0.001, type=float)
    parser.add_argument("--AUXILIARY_LR", "-alr", default=0.0001, type=float)
    parser.add_argument("--DATASET", "-data",
                        default="sst")
    parser.add_argument("--LEXICON", "-lex",
                        default="huliu_lexicon")
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")
    args = parser.parse_args()
    print(args)

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"


    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    print("Loading embeddings from {0}".format(args.EMBEDDINGS))
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
    datadir = os.path.join("../data", args.DATASET + "-" + args.FINE_GRAINED)
    sst = SSTDataset(vocab, False, datadir)

    maintask_train_iter = sst.get_split("train")
    maintask_dev_iter = sst.get_split("dev")
    maintask_test_iter = sst.get_split("test")

    maintask_loader = DataLoader(maintask_train_iter,
                                 batch_size=args.BATCH_SIZE,
                                 collate_fn=maintask_train_iter.collate_fn,
                                 shuffle=True)

    lexicon_file = os.path.join("../data", args.LEXICON)
    lexicon_dataset = LexiconDataset(vocab, False, lexicon_file)
    lex_train = lexicon_dataset.get_split("train")
    lex_test = lexicon_dataset.get_split("test")

    # Get new embedding matrix so that words not included in pretrained embeddings have a random embedding

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, args.EMBEDDING_DIM))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))


    train_model(vocab,
                new_matrix,
                len(sst.labels),
                args.EMBEDDING_DIM,
                args.HIDDEN_DIM,
                args.NUM_LAYERS,
                args.TRAIN_EMBEDDINGS,
                lex_train,
                lex_test,
                maintask_loader,
                maintask_train_iter,
                maintask_dev_iter,
                AUXILIARY_TASK=args.AUXILIARY_TASK,
                epochs=10,
                sentiment_learning_rate=args.SENTIMENT_LR,
                auxiliary_learning_rate=args.AUXILIARY_LR,
                BATCH_SIZE=50,
                number_of_runs=5,
                random_seeds=[123, 456, 789, 101112, 131415]
                )
