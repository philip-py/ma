import copy
import hashlib
import json
import logging
import os.path
import pickle
import pprint
import random
import sys
from collections import Counter
from math import fabs, log
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Set
from pydantic import BaseModel

# import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import spacy
from app.src.d00_utils.helper import (filter_spans_overlap,
                                      filter_spans_overlap_no_merge,
                                      flatten,
                                      strip_non_ascii,
                                      get_data_dir)
from gensim import corpora, models, utils
from gensim.models import KeyedVectors, Word2Vec, tfidfmodel
from gensim.models.callbacks import CallbackAny2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.utils import simple_preprocess
from germalemma import GermaLemma
from spacy import displacy
from spacy.lang.de.stop_words import STOP_WORDS
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, DocBin, Span, Token
from spacy.util import filter_spans
from spacy_sentiws import spaCySentiWS
from tqdm import tqdm
# from transformers import pipeline
from app import db
import os
from pydantic import BaseModel
from typing import List, Optional, Union, Set
from logzero import setup_logger
from app.models import Doc as Document
from pydantic import BaseModel
from app.src.d01_ana.analysis import AnalysisBase


class ConfigDiscourse(BaseModel):
    debug: bool = False
    sample: Union[int, str, List] = None
    party: Optional[str] = None
    write_bin: bool = False
    nlp_model: str = 'de_core_news_lg'
    pipeline: List[str] = ['disc']
    corpus: Optional[List[str]] = ['plenar']
    niter: int = 1
    alpha: float = 1e-1


class AnalysisDiscourse(AnalysisBase):

    def __init__(self, dir, config):
        super().__init__(dir, config)
        self.nlp = spacy.load(config.nlp_model)
        self.build_pipeline(config.pipeline)
        self.logger.info(f'init {self.__class__.__name__}\n')
        self.alpha = config.alpha
        if config.debug:
            pprint(config)

    def build_pipeline(self, pipe):
        pass

    def __call__(self, to_disk:bool = True):
        directory = Path(self.res_dir, 'emb')
        niter = self.config.niter

        print('Number of documents: {}'.format(len(self.doc_labels)))
        sentences = MyCorpus(self.nlp, self.doc_labels)
        print(f'Beginning Discourse Analysis with parameters: \n{self.config.dict()}')

        # model params
        model = Word2Vec(
        alpha=self.alpha, vector_size=100, epochs=250, seed=42, workers=8, hashfxn=hash, sorted_vocab=1, sg=1, hs=1, negative=0, sample=1e-4, min_count=40)
#         alpha=0.0025, min_alpha=0.00001, vector_size=100, epochs=500, seed=42, workers=8, hashfxn=hash, sorted_vocab=1, sg=1, hs=1, negative=0, sample=1e-4, min_count=5)
        model.build_vocab(sentences)

#         intersect
        pre = KeyedVectors.load(f'{get_data_dir()}/emb/wiki100.kv')
        res = intersect(pre, model)
        del pre

        model.wv.add_vectors(range(model.wv.vectors.shape[0]), res, replace=True)

        # ASSERT?

        total_examples = model.corpus_count
        print(f'total examples: {model.corpus_count}')

        for i in range(niter):
            try:
                seeds = random.choices(range(1_000_000), k=niter)
                seed = seeds[i]
                print(f'Seed in iteration {i}: {seed}')
                model.seed = seed
                loss_logger = LossLogger(self.config.party, i, directory, self.logger)

                print(f'Training with alpha {model.alpha}:')

                # train
                model.train(sentences, total_examples=total_examples, epochs=model.epochs, compute_loss=True, callbacks=[loss_logger])

            except EndOfTraining:
                print(f'End of Iteration: {i}')

        print(f'Discourse Analysis complete. \nResults saved in {directory}/...')



# def discourse_analysis(directory, party=None, iter=1, sample=None, **kwargs):
#     # %env PYTHONHASHSEED=0
#     sns.set_style('darkgrid')

#     logging.basicConfig(filename='w2v.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#     if directory != 'test':
#         Path(f"res_da/{directory}/").mkdir(parents=False, exist_ok=False)

#     if not os.path.isdir('res_da/' + directory):
#         print('Directory already exists.')
#         return

#     # doc_labels = load_data(party)
#     doc_labels = session.query(Tweet).all()
#     if sample:
#         doc_labels = random.sample(doc_labels, sample)

#     print('Number of documents: {}'.format(len(doc_labels)))
#     sentences = MyCorpus(doc_labels)
#     print(f'Beginning Discourse Analysis with parameters: \n{kwargs}')

#     # model params
#     model = Word2Vec(
#     alpha=0.0025, min_alpha=0.00001, vector_size=300, epochs=300, seed=42, workers=7, hashfxn=hash, sorted_vocab=1, sg=1, hs=1, negative=0, sample=1e-4, min_count=10, **kwargs)

#     model.build_vocab(sentences)

#     # intersect
#     pre = KeyedVectors.load('embeddings/wiki.model')
#     res = intersect(pre, model)
#     del pre

#     model.wv.add_vectors(range(model.wv.vectors.shape[0]), res, replace=True)

#     # ASSERT?

#     total_examples = model.corpus_count

#     for i in range(iter):
#         try:
#             seeds = random.choices(range(1_000_000), k=iter)
#             seed = seeds[i]
#             print(f'Seed in iteration {i}: {seed}')
#             model.seed = seed
#             loss_logger = LossLogger(party, i, directory)

#             # train
#             model.train(sentences, total_examples=total_examples, epochs=model.epochs, compute_loss=True, callbacks=[loss_logger])

#         except EndOfTraining:
#             print(f'End of Iteration: {i}')

#     print(f'Discourse Analysis complete. \nResults saved in {directory}/...')

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, nlp, sample):
        self.nlp = nlp
        self.docs = sample

    def __iter__(self):
        for sent in sentences_gen(self.nlp, self.docs):
            yield sent


class LossLogger(CallbackAny2Vec):
    """Get the Loss after every epoch and log it to a file"""

    def __init__(self, party, i, directory, logger):
        self.epoch = 1
        self.last_cum_loss = 0
        self.last_epoch_loss = 0
        self.losses = []
        self.best_loss = 1e15
        self.best_model = None
        self.name = party
        self.iteration = i
        self.folder = directory
        self.logger = logger

    def on_epoch_end(self, model):
        logging = self.logger

        cum_loss = model.get_latest_training_loss()
        logging.info("Cumulative Loss after epoch {}: {}".format(self.epoch, cum_loss))
        logging.info("Cumulative Loss last epoch : {}".format(self.last_cum_loss))
        this_epoch_loss = cum_loss - self.last_cum_loss
        loss_diff = this_epoch_loss - self.last_epoch_loss
        self.losses.append(this_epoch_loss)

        logging.info("Loss in epoch {}: {}".format(self.epoch, this_epoch_loss))
        logging.info("Loss in last epoch: {}".format(self.last_epoch_loss))
        logging.info("Loss difference since last epoch: {}".format(loss_diff))
        print(f"Epoch: {self.epoch} | Loss: {this_epoch_loss}")
        print(f"Loss difference: {loss_diff}")

        if this_epoch_loss < self.best_loss:
            self.best_model = model
            self.best_loss = this_epoch_loss
            logging.info(
                "saving best model in epoch {} with loss {}".format(
                    self.epoch, this_epoch_loss
                )
            )
            print("saving best model in epoch {} with loss {}".format(
                    self.epoch, this_epoch_loss))
            model.save(strip_non_ascii(f"{self.folder}/emb_{self.name}_{self.iteration}.model"))

        self.epoch = self.epoch + 1
        self.last_cum_loss = cum_loss
        self.last_epoch_loss = this_epoch_loss

        if this_epoch_loss == 0.0:
            # sys.exit()
            sns.lineplot(data=self.losses)
            plt.show()
            raise EndOfTraining()


class EndOfTraining(Exception):
    pass


def load_data(party):
    with open("data/doc_labels_plenar.pkl", "rb") as f:
        doc_labels_plenar = pickle.load(f)

    # doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]

    doc_labels = [*doc_labels_plenar]

    if party == "all":
        return doc_labels

    df = pd.read_json("data/plenar_meta.json", orient="index")
    res = df.loc[df.party == party].index.values
    doc_labels = [i.split(".txt")[0] for i in res]
    # return random.sample(doc_labels, 1)
    return doc_labels


def gendocs(label):
    with open("data/corpus_clean/{}.txt".format(label), "r") as text_file:
        return text_file.read()


def hash(w):
    return int(hashlib.md5(w.encode('utf-8')).hexdigest()[:9], 16)


def intersect(pre, new):
    """
    intersect embeddings weights
    pre -> pre-trained embeddings
    """
    res = np.zeros(new.wv.vectors.shape)
    for i, word in enumerate(new.wv.index_to_key):
        if pre.has_index_for(word):
            res[i] = pre.get_vector(word)
        else:
            res[i] = new.wv.get_vector(word)
    return res


def merge_embeddings(models):
    """
    models -> List of models from Word2Vec.load('path')
    Returns model with average weights of all embeddings
    """
    matrices = []
    for model in models:
        matrices.append(model.wv.vectors)
    matrix_merged = np.mean(np.array(matrices), axis=0)
    res = models[0]
    res.wv.add_vectors(
        range(res.wv.vectors.shape[0]), matrix_merged, replace=True)
    return res


# folder has to be fixed
def load_models(dir, party, iter):
    all_models = []
    for i in range(iter+1):
        all_models.append(Word2Vec.load(
            f'app/data/res/w2v/{dir}/emb/emb_{party}_{i}.model'))
    return all_models


def sentences_gen(nlp, labels):
    lemmatizer = GermaLemma()
#     nlp = spacy.load("de_core_news_lg")

    def lemma_getter(token):
        # if " " in token.text:
        #     return token.lemma_.lower()
        try:
            return lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    Token.set_extension('lemma', getter=lemma_getter, force=True)

    for label in labels:
        # doc = nlp(gendocs(label))
        doc = nlp(label.text)
        for i, sent in enumerate(doc.sents):
            res = []
            for j, token in enumerate(sent):
                if token.is_alpha and not token.is_punct and not token.is_digit and not token.is_space:
                    token_res = token._.lemma.lower()
                    res.append(token_res)
            res = [word for word in res if not word in STOP_WORDS]
            if len(res) <=1:
                pass
            else:
                yield res

def doc_gen(labels):
    lemmatizer = GermaLemma()
    nlp = spacy.load("de_core_news_lg")

    def lemma_getter(token):
        # if " " in token.text:
        #     return token.lemma_.lower()
        try:
            return lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    for label in labels:
        # doc = nlp(gendocs(label))
        doc = nlp(label.text)
        Token.set_extension('lemma', getter=lemma_getter, force=True)
        res = []
        for _, token in enumerate(doc):
            if token.is_alpha and not token.is_punct and not token.is_digit and not token.is_space:
                tok = token._.lemma.lower()
                tok = tok.replace('.', '')
                res.append(tok)
        res = [word for word in res if not word in STOP_WORDS]
        yield res

# def sentences_gen(labels):
#     lemmatizer = GermaLemma()
#     nlp = spacy.load("de_core_news_lg")

#     def lemma_getter(token):
#         # if " " in token.text:
#         #     return token.lemma_.lower()
#         try:
#             return lemmatizer.find_lemma(token.text, token.tag_).lower()
#         except:
#             return token.lemma_.lower()

#     for label in labels:
#         # doc = nlp(gendocs(label))
#         doc = nlp(label.text)
#         for i, sent in enumerate(doc.sents):
#             res = []
#             for j, token in enumerate(sent):
#                 Token.set_extension('lemma', getter=lemma_getter, force=True)
#                 if token.is_alpha and not token.is_punct and not token.is_digit and not token.is_space:
#                     tok = token._.lemma.lower()
#                     tok = tok.replace('.', '')
#                     res.append(tok)
#             res = [word for word in res if not word in STOP_WORDS]
#             yield res
