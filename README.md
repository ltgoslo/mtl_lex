# Multitask incorporation of lexicon information for neural sentiment classification

[Samia Touileb](samiat@ifi.uio.no),
[Jeremy Barnes](jeremycb@ifi.uio.no),
[Lilja Øvrelid](liljao@ifi.uio.no),
[Erik Velldal](erikv@ifi.uio.no)


The experiments here explore the use of multi-task learning (MTL) for incorporating external knowledge in neural models. Specifically, we show how MTL can enable a BiLSTM sentiment classifier to incorporate information from sentiment lexicons. The repo contains the models and data used for the experiments from the following paper presented at NoDaLiDa 2019:

Jeremy Barnes, Samia Touileb, Lilja Øvrelid, and Erik Velldal. 2019. **Lexicon information in neural sentiment analysis: a multi-task learning approach**. In *Proceedings of NoDaLiDa 2019*.


## Model
1. BOW: A Linear SVM trained with bag-of-words features.
2. BOW+LEX: A Linear SVM trained with bag-of-words and lexicon features.
3. STL: A BiLSTM max pooling classifer trained on the sentiment main tasks.
4. LexEmb: The same sentiment classifier, but an extra lexicon prediction model which is seperately trained to get sentiment embeddings. These embeddings are then concatenated to the word embeddings of the main model.
5. MTL: The same sentiment classifier, but with an additional lexicon prediction module which is trained jointly.

## Datasets
1. NoRec Eval (Norwegian)
2. [Stanford Sentiment Treebank](http://aclweb.org/anthology/D/D13/D13-1170.pdf) (English)


### Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision```

### How to run experiments
In order to reproduce the experiments, you will need the [pretrained embeddings](https://drive.google.com/open?id=1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS). Untar the file and set the ``` --EMBEDDINGS``` variable to point to the 'google.txt' embeddings. If you're not interested in reproducing the experiments, you can always use your own embeddings.


```
python3 model_training -h
usage: model_training.py [-h] [--NUM_LAYERS NUM_LAYERS]
                         [--HIDDEN_DIM HIDDEN_DIM] [--BATCH_SIZE BATCH_SIZE]
                         [--EMBEDDING_DIM EMBEDDING_DIM] [--TRAIN_EMBEDDINGS]
                         [--AUXILIARY_TASK AUXILIARY_TASK]
                         [--EMBEDDINGS EMBEDDINGS]
                         [--SENTIMENT_LR SENTIMENT_LR]
                         [--AUXILIARY_LR AUXILIARY_LR] [--DATASET DATASET]
                         [--LEXICON LEXICON] [--FINE_GRAINED FINE_GRAINED]

optional arguments:
  --NUM_LAYERS NUM_LAYERS, -nl NUM_LAYERS
  --HIDDEN_DIM HIDDEN_DIM, -hd HIDDEN_DIM
  --BATCH_SIZE BATCH_SIZE, -bs BATCH_SIZE
  --EMBEDDING_DIM EMBEDDING_DIM, -ed EMBEDDING_DIM
  --TRAIN_EMBEDDINGS, -te
  --AUXILIARY_TASK AUXILIARY_TASK, -aux AUXILIARY_TASK
  --EMBEDDINGS EMBEDDINGS, -emb EMBEDDINGS
  --SENTIMENT_LR SENTIMENT_LR, -slr SENTIMENT_LR
  --AUXILIARY_LR AUXILIARY_LR, -alr AUXILIARY_LR
  --DATASET DATASET, -data DATASET
  --LEXICON LEXICON, -lex LEXICON
  --FINE_GRAINED FINE_GRAINED, -fg FINE_GRAINED
                        Either 'fine' or 'binary' (defaults to 'fine'.

```

The default will run the Multi-task learning experiment on the Stanford Sentiment Treebank using the Hu and Liu lexicon as an auxiliary task. To run the single task (STL) model, just set ```--AUXILIARY_TASK none```. This will train and save the models.

In order to test the models, use test_model.py.

```
python3 test_model.py -h
usage: test_model.py [-h] [--AUXILIARY_TASK AUXILIARY_TASK]
                     [--SAVED_MODELS_DIR SAVED_MODELS_DIR]
                     [--NUM_RUNS NUM_RUNS] [--METRIC METRIC]
                     [--DATASET DATASET] [--LEXICON LEXICON]
                     [--PRINT_PREDICTIONS] [--FINE_GRAINED FINE_GRAINED]

optional arguments:
  -h, --help            show this help message and exit
  --AUXILIARY_TASK AUXILIARY_TASK, -aux AUXILIARY_TASK
  --SAVED_MODELS_DIR SAVED_MODELS_DIR, -smd SAVED_MODELS_DIR
  --NUM_RUNS NUM_RUNS, -nr NUM_RUNS
  --METRIC METRIC, -m METRIC
  --DATASET DATASET, -data DATASET
  --LEXICON LEXICON, -lex LEXICON
  --PRINT_PREDICTIONS, -pp
  --FINE_GRAINED FINE_GRAINED, -fg FINE_GRAINED
                        Either 'fine' or 'binary' (defaults to 'fine'.

```

