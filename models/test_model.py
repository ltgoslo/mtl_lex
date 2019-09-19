import argparse
import pickle

from model_training import *
from lexicon_mtl_model import *
from Utils.sst import *

def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_acc = 0.0
    best_weights = ''
    for file in os.listdir(weightdir):
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        lstm_dim = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        lstm_layers = int(re.findall('[0-9]+', file.split('-')[-2])[0])
        acc = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if acc > best_acc:
            best_params = [epochs, lstm_dim, lstm_layers]
            best_acc = acc
            weights = os.path.join(weightdir, file)
            best_weights = weights
    return best_acc, best_params, best_weights

def test_model(dataset, finegrained, aux_task, num_runs=5, metric="acc",
               saved_models_dir="saved_models", lexicon="huliu_lexicon"):

    f1s = []
    accs = []
    aux_accs = []
    preds = []
    ys = []

    print("opening model params from {0}/{1}/{2}/{3}...".format(saved_models_dir,
                                                            dataset,
                                                            finegrained,
                                                            aux_task,
                                                            lexicon))
    with open(os.path.join(saved_models_dir,
                           dataset,
                           finegrained,
                           aux_task,
                           lexicon,
                           "params.pkl"), "rb") as infile:
        params = pickle.load(infile)

    (w2idx,
     matrix_shape,
     tag_to_ix,
     len_labels,
     task2label2id) = params

    vocab = Vocab(train=False)
    vocab.update(w2idx)

    datadir = os.path.join("../data", dataset + "-" + finegrained)
    sst = SSTDataset(vocab, False, datadir)
    maintask_test_iter = sst.get_split("test")

    num_labels = len(sst.labels)

    lexicon_file = os.path.join("../data", args.LEXICON)
    lexicon_dataset = LexiconDataset(vocab, False, lexicon_file)
    lex_test = lexicon_dataset.get_split("test")

    new_matrix = np.zeros(matrix_shape)
    vocab_size, embedding_dim = matrix_shape


    print("finding best weights for runs 1 - {0}".format(num_runs))
    for i in range(num_runs):
        run = i + 1
        weight_dir = os.path.join(saved_models_dir,
                                  dataset,
                                  finegrained,
                                  aux_task,
                                  lexicon,
                                  str(run))
        best_acc, (epochs, lstm_dim, lstm_layers), best_weights =\
                                                   get_best_run(weight_dir)

        model = MTL_lexicon(vocab,
                            new_matrix,
                            num_labels,
                            2,
                            embedding_dim,
                            100,
                            num_layers=2,
                            lstm_dropout=0.2,
                            word_dropout=0.5,
                            train_embeddings=True)

        model.load_state_dict(torch.load(best_weights))
        model.eval()

        print("Run {0}".format(run))
        f1, acc, pred, x, y = model.eval_sent(maintask_test_iter, batch_size=50)
        print()

        aux_x, aux_y = zip(*lex_test)
        aux_x = torch.tensor(aux_x)
        aux_y = torch.tensor(aux_y)
        aux_acc = model.eval_aux(aux_x, aux_y)

        f1s.append(f1)
        accs.append(acc)
        aux_accs.append(aux_acc)
        preds.append(pred)
        ys.append(y)

        # print predictions to check
        if args.PRINT_PREDICTIONS:
            print("Printint predictions...")
            prediction_dir = os.path.join("predictions", dataset, finegrained,
                                          aux_task, lexicon)
            os.makedirs(prediction_dir, exist_ok=True)
            with open(os.path.join(prediction_dir, "run{0}_pred.txt".format(run)), "w") as out:
                for line in pred:
                     out.write("{0}\n".format(line))

            with open(os.path.join(prediction_dir, "gold.txt"), "w") as out:
                for line in y:
                    out.write("{0}\n".format(line))

            with open(os.path.join(prediction_dir, "sents.txt"), "w") as out:
                for l in x:
                    sent = " ".join(vocab.ids2sent([int(i) for i in l]))
                    out.write("{0}\n".format(sent))

    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    mean_aux_acc = np.mean(aux_accs)
    std_aux_acc = np.std(aux_accs)

    print("#"*20 + "FINAL" + "#"*20)

    if metric == "f1":
        print("MEAN F1: {0:.2f} ({1:.1f})".format(mean_f1 * 100, std_f1 * 100))

    if metric == "acc":
        print("MEAN ACC: {0:.2f} ({1:.1f})".format(mean_acc * 100, std_acc * 100))

    print("MEAN AUX ACC: {0:.2f} ({1:.1f})".format(mean_aux_acc * 100, std_aux_acc * 100))

    return f1s, accs, preds, ys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="lexicon_prediction")
    parser.add_argument("--SAVED_MODELS_DIR", "-smd", default="saved_models")
    parser.add_argument("--NUM_RUNS", "-nr", default=5, type=int)
    parser.add_argument("--METRIC", "-m", default="f1")
    parser.add_argument("--DATASET", "-data",
                        default="sst")
    parser.add_argument("--LEXICON", "-lex",
                        default="huliu_lexicon")
    parser.add_argument("--PRINT_PREDICTIONS", "-pp", action="store_true")
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")
    args = parser.parse_args()

    f1s, accs, preds, ys = test_model(args.DATASET,
                                      args.FINE_GRAINED,
                                      args.AUXILIARY_TASK,
                                      num_runs=args.NUM_RUNS,
                                      metric=args.METRIC,
                                      saved_models_dir=args.SAVED_MODELS_DIR,
                                      lexicon=args.LEXICON)

