import os
import copy
import pickle
from pprint import pprint
import random
import spacy
from collections import Counter
from datetime import datetime
from math import fabs, log
from pathlib import Path
from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd
from app import db, config as app_config
from app.models import Doc as Document
from app.src.d00_utils.helper import (filter_spans_overlap)
from germalemma import GermaLemma
from logzero import setup_logger
from pydantic import BaseModel
from spacy.lang.de.stop_words import STOP_WORDS
from spacy_sentiws import spaCySentiWS
from tqdm import tqdm

from .clf import Clf
from .pipeline import *
# from .diskurs import MyCorpus

class ConfigDiscourse(BaseModel):
    debug: bool = False
    sample: Union[int, str, List] = None
    party: Optional[str] = None
    window_size: int = 25
    write_bin: bool = False
    nlp_model: str = 'de_core_news_lg'
    pipeline: List[str] = ['disc']
    corpus: Optional[List[str]] = ['plenar']
    niter: int = 1


class Config(BaseModel):
    debug: bool = False
    sample: Union[int, str, List] = None
    party: Optional[str] = None
    window_size: int = 25
    write_bin: bool = False
    write_arr: bool = False
    to_db: bool = False
    nlp_model: str = 'de_core_news_lg'
    clf_model: Optional[str] = None
    pipeline: List[str] = ['extensions', 'sentiment', 'entity', 'res', 'spans', 'clf']
    corpus: Optional[List[str]] = None


class AnalysisBase:
    """Base-Class for Elements of Pipeline"""

    def __init__(self, dir, config):
        # self.res_dir = Path('res', dir)
        self.res_dir = os.path.join(app_config[os.getenv('FLASK_CONFIG')].DIR_RES, dir)
        self.config = config
#         self.logfile = app.config.get('LOGFILE')
#         self.logfile = os.environ.get('FLASK_CONFIG')
        self.logfile = 'logfile_dev.log'
        self.logger = setup_logger(
            logfile=self.logfile, disableStderrLogger=True)
        self.doc_labels, self.input = self.load_data(config.sample)

        if dir != 'test' and os.getenv('FLASK_CONFIG') != 'testing':
            Path(f"{self.res_dir}/docs").mkdir(parents=True, exist_ok=False)
            Path(f"{self.res_dir}/emb").mkdir(parents=True, exist_ok=False)



    def load_data(self, sample: Union[int, str, List]) -> tuple:
        """load data from db or set input as only document"""
        text = None
        if type(sample) == int:
            if self.config.corpus:
                doc_labels = db.session.query(Document).filter(
                    Document.corpus.in_(self.config.corpus)).all()
                # doc_labels = Document.query.filter(Document.corpus.in_(self.corpus.config)).all()
            else:
                doc_labels = db.session.query(Document).all()
            doc_labels = random.sample(doc_labels, sample)
        elif type(sample) == str:
            doc_labels = ['input']
            text = sample
        elif type(sample) == list:
            doc_labels = sample
        else:
            if self.config.corpus:
                doc_labels = db.session.query(Document).filter(
                    Document.corpus.in_(self.config.corpus)).all()
                # doc_labels = Document.query.filter_by(Document.corpus.in_(self.corpus.config)).all()
            else:
                doc_labels = db.session.query(Document).all()

        if self.config.party:
            doc_labels = [i for i in doc_labels if i.autor.party == self.config.party]
#             better:
#             doc_labels = Document.query.filter(Document.autor.has(Akteur.party == self.config.party))
        return (doc_labels, text)


class Analysis(AnalysisBase):
    """Class to create Pipeline for all linguistics Annotations on Tokens, Spans, Entities and Docs.
    """

    def __init__(self, dir, config):
        super().__init__(dir, config)
        self.nlp = spacy.load(config.nlp_model)
        self.build_pipeline(config.pipeline)
        self.logger.info(f'init {self.__class__.__name__}\n')
        if config.debug:
            pprint(config)

    def build_pipeline(self, pipe: List[str]) -> None:
        """Initiate Pipeline-Elements and add them to nlp"""
        self.nlp.remove_pipe("ner")
        if 'extensions' in pipe:
            extensions = CustomExtensions(self.nlp, self.doc_labels)
            self.nlp.add_pipe(extensions, last=True)
        if 'sentiment' in pipe:
            sentiment = SentimentRecognizer(self.nlp)
            sentiws = spaCySentiWS(sentiws_path=f"{config[os.getenv('FLASK_CONFIG')].DIR_DATA}/sentiws/")
            self.nlp.add_pipe(sentiment, last=True)
            self.nlp.add_pipe(sentiws, last=True)
        if 'entity' in pipe:
            entity = EntityRecognizer(self.nlp)
            self.nlp.add_pipe(entity, last=True)
        if 'res' in pipe:
            res = ContentAnalysis(self.nlp)
            self.nlp.add_pipe(res, last=True)
        if 'spans' in pipe:
            spans = Spans(self.nlp)
            self.nlp.add_pipe(spans, last=True)
        if 'clf' in pipe:
            clf = Clf(self.nlp, self.config)
            self.nlp.add_pipe(clf, last=True)

    def __call__(self, to_disk: bool = False, to_db: bool = False):
        """Runs the analysis"""
        self.logger.info(
            "Number of documents: {}".format(len(self.doc_labels)))
        self.logger.info(
            f"Beginning Content Analysis with parameters: \n {self.config.dict()}")
        self.is_test = os.getenv('FLASK_CONFIG') == 'testing'
        self.is_res = True
        # self.is_res = "res" in self.nlp.pipeline
        # if self.is_res:
        res = self.get_results()

        if to_disk:
            self.config.write_bin = True
            self.config.write_arr = True

        if self.config.to_db:
            to_db = True

        matrices = []
        # labels = []


        # for document in tqdm(self.doc_labels):
        for document in (self.doc_labels if self.config.debug else tqdm(self.doc_labels)):
            # if self.is_res:
            res.labels.append(document.id)
            if self.input:
                doc = self.nlp(self.input)
                if self.config.debug:
                    self.debug_doc()
            else:
                doc = self.nlp(document.text)
                if self.config.write_arr:
                    arr = doc.to_array(['orth', 'lower', 'lemma'])
                    matrices.append(arr)
                if to_db:
                    arr = doc.to_array(['orth', 'lower', 'lemma'])
                    self.arr_to_db(arr, document)
                # if self.config.write_bin:
                #     bin = doc.to_bytes()
                #     self.bin_to_db(bin, document)
                # if to_db:
                #     bin = doc.to_bytes()
                #     self.bin_to_disk(bin, document)
                #res to db!
                if self.is_res:
                    document.res = res.viz[-1]

        if self.config.write_arr:
            self.arr_to_disk(matrices)
        if to_disk:
            self.res_to_disk()
        if to_db & ~self.is_test:
            db.session.commit()

        self.logger.info(
            f"\nContent Analysis complete. \nResults saved in {self.res_dir}/")

    def bin_to_db(self, bin, document):
        document.bin = bin
        # db.session.add()

    def arr_to_db(self, arr, document):
        document.arr = arr
        # db.session.add()

    def bin_to_disk(self, bin, document):
        with open(Path(self.res_dir, 'docs', str(document.id)), 'wb') as f:
            f.write(bin)

    def arr_to_disk(self, matrices):
        matrices = np.array(matrices)
        if ~self.is_test:
            np.save(Path(self.res_dir, 'matrix.np'), matrices)
            with open(f'{self.res_dir}/labels.pkl', 'wb') as f:
                pickle.dump(self.get_results().labels, f)
            self.nlp.vocab.to_disk(f'{self.res_dir}/vocab')

    def res_to_disk(self):
        with open(f'{self.res_dir}/results_all.pkl', 'wb') as f:
            pickle.dump(self.get_results(), f)

    def debug_doc(self):
        for token in doc:
            print(token.text, token.ent_type_, token._.is_elite_neg,
                  token._.is_attr, token._.is_negated, 'lemma', token._.lemma)

    def get_results(self, n: int = -1):
        # for element in self.nlp.pipeline:
        #     if element[0] == 'content_analysis':
        #         res = element[1]
        # return res.results
        return self.nlp.pipeline[n][1].results

    def to_db(self):
        labels = pickle.load(open(f"{self.res_dir}/labels.pkl", "rb"))
        documents = QUERYlabels
        res_all = pickle.load(open(f'{self.res_dir}/results_all.pkl'))
        arrs = np.load(f'{res_dir}/matrix.np.npy')
        docs = glob(f'{res_dir}/docs/*')

        for i, doc in enumerate(documents):
            res = res_all[i]
            arr = arrs[i]
            with open(docs[i], 'rb') as f:
                doc_bytes = f.read()
            doc.res = res
            doc.arr = arr
            doc.bin = doc_bytes
            # db.session.add()
        db.session.commit()


# class Config(BaseModel):
#     debug: bool = False
#     sample: Union[int, str, List] = None
#     party: Optional[str] = None
#     window_size: int = 25
#     write_bin: bool = False
#     write_arr: bool = False
#     to_db: bool = False
#     nlp_model: str = 'de_core_news_lg'
#     clf_model: Optional[str] = None
#     pipeline: List[str] = ['extensions', 'sentiment', 'entity', 'res', 'clf']
#     corpus: Optional[List[str]] = None


# class AnalysisBase:
#     """Base-Class for Elements of Pipeline"""

#     def __init__(self, dir, config):
#         self.res_dir = Path('res', dir)
#         self.config = config
# #         self.logfile = app.config.get('LOGFILE')
# #         self.logfile = os.environ.get('FLASK_CONFIG')
#         self.logfile = 'logfile_dev.log'
#         self.logger = setup_logger(
#             logfile=self.logfile, disableStderrLogger=True)
#         self.doc_labels, self.input = self.load_data(config.sample)

#         if dir != 'test':
#             Path(f"{self.res_dir}/docs").mkdir(parents=True, exist_ok=False)

#     def load_data(self, sample: Union[int, str, List]) -> tuple:
#         """load data from db or set input as only document"""
#         text = None
#         if type(sample) == int:
#             if self.config.corpus:
#                 doc_labels = db.session.query(Document).filter(
#                     Document.corpus.in_(self.config.corpus)).all()
#                 # doc_labels = Document.query.filter(Document.corpus.in_(self.corpus.config)).all()
#             else:
#                 doc_labels = db.session.query(Document).all()
#             doc_labels = random.sample(doc_labels, sample)
#         elif type(sample) == str:
#             doc_labels = ['input']
#             text = sample
#         elif type(sample) == list:
#             doc_labels = sample
#         else:
#             if self.config.corpus:
#                 doc_labels = db.session.query(Document).filter(
#                     Document.corpus.in_(self.config.corpus)).all()
#                 # doc_labels = Document.query.filter_by(Document.corpus.in_(self.corpus.config)).all()
#             else:
#                 doc_labels = db.session.query(Document).all()
#         return (doc_labels, text)


# class Analysis(AnalysisBase):
#     """Class to create Pipeline for all linguistics Annotations on Tokens, Spans, Entities and Docs.
#     """

#     def __init__(self, dir, config):
#         super().__init__(dir, config)
#         self.nlp = spacy.load(config.nlp_model)
#         self.build_pipeline(config.pipeline)
#         self.logger.info(f'init {self.__class__.__name__}\n')

#     def build_pipeline(self, pipe: List[str]) -> None:
#         """Initiate Pipeline-Elements and add them to nlp"""
#         self.nlp.remove_pipe("ner")
#         if 'extensions' in pipe:
#             extensions = CustomExtensions(self.nlp, self.doc_labels)
#             self.nlp.add_pipe(extensions, last=True)
#         if 'sentiment' in pipe:
#             sentiment = SentimentRecognizer(self.nlp)
#             sentiws = spaCySentiWS(sentiws_path='sentiws/')
#             self.nlp.add_pipe(sentiment, last=True)
#             self.nlp.add_pipe(sentiws, last=True)
#         if 'entity' in pipe:
#             entity = EntityRecognizer(self.nlp)
#             self.nlp.add_pipe(entity, last=True)
#         if 'res' in pipe:
#             res = ContentAnalysis(self.nlp)
#             self.nlp.add_pipe(res, last=True)
#         if 'spans' in pipe:
#             spans = Spans(self.nlp)
#             self.nlp.add_pipe(spans, last=True)
#         if 'clf' in pipe:
#             clf = Clf(self.nlp, self.config)
#             self.nlp.add_pipe(clf, last=True)

#     def __call__(self, to_disk: bool = False, to_db: bool = False):
#         """Runs the analysis"""
#         self.logger.info(
#             "Number of documents: {}".format(len(self.doc_labels)))
#         self.logger.info(
#             f"Beginning Content Analysis with parameters: \n {self.config.dict()}")
#         res = self.get_results()

#         if to_disk:
#             self.config.write_bin = True
#             self.config.write_arr = True

#         if self.config.to_db:
#             to_db = True

#         matrices = []
#         # labels = []

#         for document in tqdm(self.doc_labels):
#             res.labels.append(document.id)
#             if self.input:
#                 doc = self.nlp(self.input)
#                 if self.config.debug:
#                     self.debug_doc()
#             else:
#                 doc = self.nlp(document.text)
#                 if self.config.write_arr:
#                     arr = doc.to_array(['orth', 'lower', 'lemma'])
#                     self.arr_to_db(arr, document)
#                     matrices.append(arr)
#                 if self.config.write_bin:
#                     bin = doc.to_bytes()
#                     self.bin_to_db(bin, document)
#                     self.bin_to_disk(bin, document)
#                 document.res = res.viz[-1]

#         if self.config.write_arr:
#             self.arr_to_disk(matrices)
# #         res.prepare()
#         self.res_to_disk()
#         if to_db:
#             db.session.commit()

#         self.logger.info(
#             f"\nContent Analysis complete. \nResults saved in {self.res_dir}/")

#     def bin_to_db(self, bin, document):
#         document.bin = bin
#         # db.session.add()

#     def arr_to_db(self, arr, document):
#         document.arr = arr
#         # db.session.add()

#     def bin_to_disk(self, bin, document):
#         with open(Path(self.res_dir, 'docs', str(document.id)), 'wb') as f:
#             f.write(bin)

#     def arr_to_disk(self, matrices):
#         matrices = np.array(matrices)
#         np.save(Path(self.res_dir, 'matrix.np'), matrices)
#         with open(f'{self.res_dir}/labels.pkl', 'wb') as f:
#             pickle.dump(self.get_results().labels, f)
#         self.nlp.vocab.to_disk(f'{self.res_dir}/vocab.txt')

#     def res_to_disk(self):
#         with open(f'{self.res_dir}/results_all.pkl', 'wb') as f:
#             pickle.dump(self.get_results(), f)

#     def debug_doc(self):
#         for token in doc:
#             print(token.text, token.ent_type_, token._.is_elite_neg,
#                   token._.is_attr, token._.is_negated, 'lemma', token._.lemma)

#     def get_results(self, n: int = -1):
#         # for element in self.nlp.pipeline:
#         #     if element[0] == 'content_analysis':
#         #         res = element[1]
#         # return res.results
#         return self.nlp.pipeline[n][1].results

#     def to_db(self):
#         labels = pickle.load(open(f"{self.res_dir}/labels.pkl", "rb"))
#         documents = QUERYlabels
#         res_all = pickle.load(open(f'{self.res_dir}/results_all.pkl'))
#         arrs = np.load(f'{res_dir}/matrix.np.npy')
#         docs = glob(f'{res_dir}/docs/*')

#         for i, doc in enumerate(documents):
#             res = res_all[i]
#             arr = arrs[i]
#             with open(docs[i], 'rb') as f:
#                 doc_bytes = f.read()
#             doc.res = res
#             doc.arr = arr
#             doc.bin = doc_bytes
#             # db.session.add()
#         db.session.commit()

def recount_viz(df, dictionary, idf_weight):
    df['viz'] = df.apply(lambda row: ContentAnalysis.recount_viz(
        row['viz'], row['doclen'], dictionary, idf_weight), axis=1)




def serialize(directory, party='all', sample=None):

    Path(f"nlp/{directory}/docs").mkdir(parents=True, exist_ok=False)

    doc_labels = load_data(party)
    if type(sample) == int:
        doc_labels = random.sample(doc_labels, sample)
        text = None
    elif type(sample) == str:
        doc_labels = ['test']
        text = sample
    elif type(sample) == list:
        doc_labels = sample
        text = None
    else:
        text = None
    print("Number of documents: {}".format(len(doc_labels)))
    print(f"Beginning Serialization with parameters: \n Party: {party}")
    nlp = spacy.load("de_core_news_lg")
    ca = ContentAnalysis(nlp)
    entity_recognizer = EntityRecognizer(nlp)
    sentiment_recognizer = SentimentRecognizer(nlp)
    # nlp.add_pipe(ca, last=True)
    nlp.add_pipe(custom_lemma, last=True)
    nlp.add_pipe(sentiment_recognizer, last=True)
    nlp.add_pipe(entity_recognizer, last=True)
    nlp.remove_pipe("ner")
    labels = []
    # doc_bin = DocBin(attrs=["LEMMA", "POS", "DEP", "ENT_TYPE"], store_user_data=True)
    for label in tqdm(doc_labels):
        labels.append(label)
        if text:
            doc = nlp(text)

        else:
            doc = nlp(gendocs(label))
        # json_doc = doc.to_json(['has_volk', 'has_elite'])
        # doc_bin.add(doc)
        # with open(f'nlp/test/{label}.json', 'w') as outfile:
        #     json.dump(json_doc, outfile)
        doc_bytes = doc.to_bytes()
        with open(f'nlp/{directory}/docs/{label}', 'wb') as f:
            f.write(doc_bytes)
    # nlp.to_disk('nlp/test')
    # data = doc_bin.to_bytes()
    # with open(f'nlp/{directory}/docs_plenar', 'wb') as f:
    #     f.write(data)
    nlp.vocab.to_disk(f'nlp/{directory}/vocab.txt')
    with open(f'nlp/{directory}/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    print(f"Serialization complete. \nResults saved in nlp/{directory}/")




def lemma_getter_standalone(token):
    lemmatizer = GermaLemma()
    # if " " in token.text:
    #     return token.lemma_.lower()
    try:
        return lemmatizer.find_lemma(token.text, token.tag_).lower()
    except:
        return token.lemma_.lower()

# def find_lexeme(array):
#     res = []
#     for a in array:
#         res.append(np.where(a == nlp.vocab['Danke'].orth)[0])
#     return res

# def search_sequence_numpy(arr,seq):
#     """ Find sequence in an array using NumPy only.

#     Parameters
#     ----------
#     arr    : input 1D array
#     seq    : input 1D array

#     Output
#     ------
#     Output : 1D Array of indices in the input array that satisfy the
#     matching of input sequence in the input array.
#     In case of no match, an empty list is returned.
#     """

#     # Store sizes of input array and sequence
#     Na, Nseq = arr.size, seq.size

#     # Range of sequence
#     r_seq = np.arange(Nseq)

#     # Create a 2D array of sliding indices across the entire length of input array.
#     # Match up with the input sequence & get the matching starting indices.
#     M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

#     # Get the range of those indices as final output
#     if M.any() >0:
#         return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
#     else:
#         return []         # No match found

# def find_sequence(matrix, sequence_string, nlp):
#     sequence_int = np.array([nlp.vocab[i] for i in sequence_string.split()])
#     res = []
#     for doc in matrix:
#         a = doc[:,0]
#         hits = search_sequence_numpy(a, sequence_int)
#         res.append(hits)
#     return res




# def render(text, row, viz, span=None, filter_by=['score'], pres=False, online=False):
#     """visualize documents with displacy"""

#     def filter_by_condition(viz, condition):
#         viz = [i for i in viz if i[condition]]
#         return viz

#     viz = filter_viz(viz, on='start')
#     viz = filter_spans_overlap(viz)
#     viz_span = []

#     if span:
#         span = span
#     else:
#         span = (0, len(text) + 1)

#     if pres:
#         viz_span_ = []
#         for hit in viz:
#             paragraph = {}
#             hit['start'] -= span[0]
#             hit['end'] -= span[0]
#             paragraph['start'] = hit['span_start']
#             paragraph['end'] = hit['span_end']
#             # hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
#             if paragraph['start'] not in [i['start'] for i in viz_span_]:
#                 viz_span_.append(paragraph)

#         for n, v in enumerate(viz_span_):
#             viz_span.append({'start': v['start'], 'end': v['end'], 'label': f'P|{n+1}'})

#         viz_span = sorted(viz_span, key=lambda x: x['start'])

#     ##################################################
#     else:

#         if filter_by:
#             for condition in filter_by:
#                 viz = filter_by_condition(viz, condition)

#         if span[0] > 0:
#             viz = [i for i in viz if i['span_start'] == span[0]]

#         for hit in viz:

#             hit['start'] -= span[0]
#             hit['end'] -= span[0]

#             hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
#             viz_span.append(hit)

#         viz_starts = set([i['span_start'] for i in viz])

#         for n, start in enumerate(sorted(viz_starts)):
#             if start > 0 and span[0] == 0:
#                 viz_span.append({'start': start-1, 'end': start, 'label': f'P{n+1} | {start}'})

#         viz_span = sorted(viz_span, key=lambda x: x['start'])
#     ###############################################

#     if online:
#         ex = [
#             {
#                 "text": text[span[0]: span[1]],
#                 "ents": viz_span,
#                 "title": 'user-input analysis'
#                 # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
#                 # 'title': 'text'
#             }
#         ]

#     else:
#         ex = [
#             {
#                 "text": text[span[0]: span[1]],
#                 "ents": viz_span,
#                 "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
#                 # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
#                 # 'title': 'text'
#             }
#         ]
#     all_ents = {i["label"] for i in viz_span}

#     options = {"ents": all_ents, "colors": dict()}
#     for ent in all_ents:
#         if ent.startswith("E"):
#             options["colors"][ent] = "coral"
#         if ent.startswith("V"):
#             options["colors"][ent] = "lightgrey"
#         if ent.startswith("P"):
#             options["colors"][ent] = "yellow"
#     print(viz_span)
#     return (ex, options)


# class Spans(object):
#     name = "spans"

#     def __init__(self, nlp):
#         self.nlp = nlp
#         self.results = self.nlp.pipeline[-1][1].results
#         self.results.spans = []
#         self.index = 0

#     def __call__(self, doc):
#         spans = []
#         for hit in self.results.viz[self.index]:
#             span_start = hit.span_start
#             span_end = hit.span_end
#             span_id = (span_start, span_end)
#             if span_id not in spans:
#                 spans.append(span_id)
#         self.results.spans.append(spans)
#         self.index += 1
#         return doc


#             hit = hit.dict()
#             if hit['start'] not in seen:
#                 span_start = hit['span_start']
#                 span_end = hit['span_end']
#                 span_id = (span_start, span_end)
#                 if span_id not in spans:
#                     # span_dict[label][span_id] = 0.0
#                     span = {span_id: 0.0}
#                 span[span_id] += hit['score']
#                 spans.append(span)
#                 seen.add(hit['start'])

#         self.results.spans.append(spans)


# class SentimentRecognizer(object):

#     name = "sentiment_recognizer"

#     def __init__(self, nlp):
#         self.load_dicts()
#         # Token.set_extension('is_neg', default=False, force=True)
#         # Token.set_extension('is_pos', default=False, force=True)
#         Token.set_extension("is_neg", getter=self.is_neg_getter, force=True)
#         Token.set_extension("is_pos", getter=self.is_pos_getter, force=True)
#         Token.set_extension("is_negated", getter=self.is_negated_getter, force=True)
#         Token.set_extension("span_sent", default=None, force=True)
#         Doc.set_extension("has_neg", getter=self.has_neg, force=True)
#         Doc.set_extension("has_pos", getter=self.has_pos, force=True)
#         Span.set_extension("has_neg", getter=self.has_neg, force=True)
#         Span.set_extension("has_pos", getter=self.has_pos, force=True)

#     def __call__(self, doc):
#         return doc

#     def is_neg_getter(self, token):
#         if token._.lemma in self.negativ:
#             if token._.is_negated:
#                 return False
#             else:
#                 return True
#         if token._.lemma in self.positiv:
#             if token._.is_negated:
#                 return True
#             else:
#                 return False

#     def is_pos_getter(self, token):
#         if token._.lemma in self.positiv:
#             if token._.is_negated:
#                 return False
#             else:
#                 return True
#         if token._.lemma in self.negativ:
#             if token._.is_negated:
#                 return True
#             else:
#                 return False

#     def is_negated_getter(self, token):

#         check = list(token.children)
#         node = token
#         # CAREFUL HERE
#         if token.pos_ == "ADJ" or token.pos_ == "ADV":
#             if token.i - 1 >= 0:
#                 check.append(token.doc[token.i - 1])
#         ####################
#         while node.head:
#             seen = node
#             if seen == node.head:
#                 # CAREFUL HERE
#                 check.extend(list(node.head.lefts))
#                 break
#             check.append(node)
#             check.extend(list(node.children))
#             if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
#                 check.append(node.head)
#                 # CAREFUL HERE #####
#                 if node.head.dep_ == 'root':
#                     check.exten(list(node.head.lefts))
#                 ####################
#                 break
#             else:
#                 node = node.head
#         attr = [
#             # child for child in check if child.dep_ == "ng" or child._.lemma in negation_words
#             child for child in check if child.dep_ == "ng" or child._.is_negation
#         ]
#         if attr:
#             return True
#         else:
#             return False

#     def load_dicts(self):
#         dict_folder = "dict"
#         sent = pd.read_csv(f"{dict_folder}/SentDict.csv")
#         self.positiv = set([
#                 x.strip()
#                 for x in sent.loc[sent.sentiment == 1, ["feature"]]["feature"].tolist()
#         ])
#         self.negativ = set([
#                 x.strip()
#                 for x in sent.loc[sent.sentiment == -1, ["feature"]]["feature"].tolist()
#         ])

#     def has_neg(self, tokens):
#         return any([t._.get("is_neg") for t in tokens])

#     def has_pos(self, tokens):
#         return any([t._.get("is_pos") for t in tokens])


# class EntityRecognizer(object):

#     name = "entity_recognizer"

#     def __init__(self, nlp):
#         self.load_dicts()
#         self.ruler = EntityRuler(nlp, overwrite_ents=True, phrase_matcher_attr="LOWER")
#         self.vocab = nlp.vocab
#         patterns = []
#         for term in self.dict_people:
#             patterns.append({"label": "PEOPLE", "pattern": [{"_": {"lemma": term}}]})
#         for term in self.dict_elite:
#             patterns.append({"label": "ELITE", "pattern": [{"_": {"lemma": term}}]})
#         for term in self.dict_elite_standalone:
#             patterns.append(
#                 {"label": "ELITE_STANDALONE", "pattern": [{"_": {"lemma": term}}]}
#             )
#         for term in self.dict_people_ord:
#             patterns.append(
#                 {"label": "PEOPLE_ORD", "pattern": [{"_": {"lemma": term}}]}
#             )
#         for term in self.dict_people_ger:
#             patterns.append(
#                 {"label": "PEOPLE_GER", "pattern": [{"_": {"lemma": term}}]}
#             )
#         for term in self.dict_attr_ord:
#             patterns.append({"label": "ATTR_ORD", "pattern": [{"_": {"lemma": term}}]})
#         for term in self.dict_attr_ger:
#             patterns.append({"label": "ATTR_GER", "pattern": [{"_": {"lemma": term}}]})
#         self.ruler.add_patterns(patterns)
#         # self.ruler.add_patterns([{'label': 'ELITE', 'pattern': 'europäische union'}])

#         Token.set_extension("is_volk", default=False, force=True)
#         Token.set_extension("is_elite", default=False, force=True)
#         Token.set_extension("is_elite_neg", default=False, force=True)
#         Token.set_extension("is_attr", default=False, force=True)
#         Token.set_extension("attr_of", default=None, force=True)
#         Doc.set_extension("has_volk", getter=self.has_volk, force=True)
#         Doc.set_extension("has_elite", getter=self.has_elite, force=True)
#         Span.set_extension("has_volk", getter=self.has_volk, force=True)
#         Span.set_extension("has_elite", getter=self.has_elite, force=True)

#     def __call__(self, doc):

#         matches = self.ruler.matcher(doc)
#         # matches.extend(self.ruler.phrase_matcher(doc))
#         spans = []
#         for id, start, end in matches:
#             entity = Span(doc, start, end, label=self.vocab.strings[id])
#             spans.append(entity)
#         filtered = filter_spans(spans)
#         for entity in filtered:
#             # People setter
#             if entity.label_ == "PEOPLE":
#                 for token in entity:
#                     token._.set("is_volk", True)
#             if entity.label_ == "PEOPLE_ORD":
#                 for token in entity:
#                     check = list(token.children)
#                     attr = set(
#                         [
#                             child
#                             for child in check
#                             if child._.lemma.lower() in self.dict_attr_ord
#                         ]
#                     )
#                     if attr:
#                         token._.set("is_volk", True)
#                         for child in attr:
#                             child._.set("is_volk", True)
#                             child._.set("is_attr", True)
#                             child._.set("attr_of", token.idx)

#             if entity.label_ == "PEOPLE_GER" or entity.label_ == "PEOPLE_ORD":
#                 for token in entity:
#                     check = list(token.children)
#                     attr = set(
#                         [
#                             child
#                             for child in check
#                             if child._.lemma.lower() in self.dict_attr_ger
#                         ]
#                     )
#                     if attr:
#                         token._.set("is_volk", True)
#                         for child in attr:
#                             child._.set("is_volk", True)
#                             child._.set("is_attr", True)
#                             child._.set("attr_of", token.idx)
#             # Elite setter
#             if entity.label_ == "ELITE":
#                 for token in entity:
#                     token._.set("is_elite", True)

#                     check = list(token.children)
#                     node = token
#                     while node.head:
#                         seen = node
#                         for t in node.children:
#                             if t.dep_ == "conj":
#                                 break
#                             check.append(t)
#                             # for tok in t.children:
#                             # #     check.append(tok)
#                             #     if tok.dep_ == "pd":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "mo":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "oa":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "oa2":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "og":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "da":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "op":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == "cc":
#                             #         check.append(tok)
#                             #     elif tok.dep_ == 'avc':
#                             #         check.append(tok)
#                             #     elif tok.dep_ == 'app':
#                             #         check.append(tok)
#                             #     elif tok.dep_ == 'adc':
#                             #         check.append(tok)
#                             #     elif tok.dep_ == 'ag':
#                             #         check.append(tok)
#                         check.append(node)
#                         # check.extend(list(node.children))
#                         if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
#                             check.append(node.head)
#                             break
#                         # if node.head.pos_ == 'CCONJ' and node.head.text in negation_cconj:
#                         if node.head.pos_ == 'CCONJ' and node.head._.is_sentence_break:
#                             check.append(node.head)
#                             break
#                         if seen == node.head:
#                             break
#                         else:
#                             node = node.head
#                     attr = set([child for child in check if child._.is_neg])
#                     if attr:
#                         token._.set("is_elite_neg", True)
#                         for child in attr:
#                             child._.set("is_elite_neg", True)
#                             child._.set("is_attr", True)
#                             child._.set("attr_of", token.idx)

#             # if entity.label_ == "ELITE" or entity.label_ == "ELITE_STANDALONE":
#             if entity.label_ == "ELITE_STANDALONE":
#                 for token in entity:
#                     token._.set("is_elite", True)
#                     if not token._.is_negated:
#                         token._.set("is_elite_neg", True)
#             doc.ents = list(doc.ents) + [entity]
#         # nach content analyse?
#         # for span in filtered:
#         # span.merge()
#         return doc

#     def load_dicts(self):
#         dict_folder = "dict"
#         # import all dicts
#         # elite
#         df_elite = pd.read_csv(f"{dict_folder}/elite_dict.csv")
#         self.dict_elite = set(
#             df_elite[df_elite.type != "elite_noneg"]["feature"].tolist()
#         )
#         self.dict_elite_standalone = set(
#             df_elite[df_elite.type == "elite_noneg"]["feature"].tolist()
#         )

#         # people
#         df_people = pd.read_csv(f"{dict_folder}/people_dict.csv")
#         self.dict_people = set(
#             df_people[df_people.type == "people"]["feature"].tolist()
#         )
#         self.dict_people_ord = set(
#             df_people[df_people.type == "people_ordinary"]["feature"].tolist()
#         )
#         self.dict_attr_ord = set(
#             df_people[df_people.type == "attr_ordinary"]["feature"].tolist()
#         )
#         self.dict_people_ger = set(
#             df_people[df_people.type == "people_ger"]["feature"].tolist()
#         )
#         self.dict_attr_ger = set(
#             df_people[df_people.type == "attr_ger"]["feature"].tolist()
#         )

#         # testing:
#         # self.dict_people.add("wir sind das volk")
#         # self.dict_elite.add("europäische union")


#     # getters
#     def has_volk(self, tokens):
#         return any([t._.get("is_volk") for t in tokens])

#     def has_elite(self, tokens):
#         return any([t._.get("is_elite") for t in tokens])


# class ContentAnalysis(object):
#     "Runs Content Analysis as spacy-pipeline-component"
#     name = "content_analysis"

#     def __init__(self, nlp, window_size=25):
#         self.nlp = nlp
#         self.dictionary = pickle.load(open("data/plenar_dict.pkl", "rb"))
#         # self.dictionary = None
#         # Results()
#         # self.res = []
#         self.results = Results()
#         self.window_size = window_size

#         Span.set_extension(
#             "has_elite_neg", getter=self.has_elite_neg_getter, force=True
#         )
#         Span.set_extension(
#             "has_volk", getter=self.has_volk_getter, force=True
#         )

#     def __call__(self, doc):
#         res = {
#             "viz": [],
#             "volk": [],
#             "volk_attr": [],
#             "elite": [],
#             "elite_attr": [],
#         }

#         ##########################################
#         window_size = self.window_size
#         # idf_weight = 1.0
#         ##########################################

#         matcher = Matcher(self.nlp.vocab)
#         pattern = [{"_": {"is_elite_neg": True}}]
#         matcher.add("text", None, pattern)
#         matches = matcher(doc)
#         doclen = len(doc)

#         # spans = set()
#         spans = []
#         token_ids = set()
#         ps_counter = 1
#         last_start = None
#         for id, start, end in matches:
#             if start - window_size < 0:
#                 start = 0
#             else:
#                 start = start - window_size
#             if end + window_size > doclen:
#                 end = doclen
#             else:
#                 end = end + window_size
#             sentence_start = doc[start].sent.start
#             sentence_end = doc[end-1].sent.end
#             # span = doc[start - window_size : end + window_size]
#             span = {'span_start': sentence_start, 'span_end': sentence_end}
#             spans.append(span)

#             """keep
#             span = doc[sentence_start : sentence_end]
#             spans.add(span)
#             """

#         # CAREFUL!!!!!
#         spans = filter_spans_overlap_no_merge(spans)
#         for span in spans:
#             span = doc[span['span_start'] : span['span_end']]
#             if span._.has_elite_neg and span._.has_volk:
#                 # check sentiment of span mit sentiws
#                 span_sentiment = sum([token._.sentiws for token in span if token._.sentiws])
#                 # if span_sentiment > 0.0:
#                 #     pass
#                 # else:
#                 for token in span:
#                     token._.span_sent = span_sentiment
#                     if token._.is_volk:
#                         # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "V", idf_weight, dictionary=self.dictionary))
#                         if token._.is_attr and token.i not in token_ids:
#                             res["volk_attr"].append(token._.lemma)
#                             res['viz'].append(self.on_hit(token, 'VA', doc[span.start], doc[span.end-1]))
#                             token_ids.add((token.i, "VA"))
#                         else:
#                             if token.i not in token_ids:
#                                 res["volk"].append(token._.lemma)
#                                 res['viz'].append(self.on_hit(token, 'V', doc[span.start], doc[span.end-1]))
#                                 token_ids.add((token.i, "V"))

#                     if token._.is_elite_neg:
#                         # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "E", idf_weight, dictionary=self.dictionary))
#                         if token._.is_attr and token.i not in token_ids:
#                             res["elite_attr"].append(token._.lemma)
#                             res['viz'].append(self.on_hit(token, 'EA', doc[span.start], doc[span.end-1]))
#                             token_ids.add((token.i, "EA"))
#                         else:
#                             if token.i not in token_ids:
#                                 res["elite"].append(token._.lemma)
#                                 res['viz'].append(self.on_hit(token, 'E', doc[span.start], doc[span.end-1]))
#                                 token_ids.add((token.i, "E"))

#         # sorts by start AND deletes duplicates!
#         res["viz"] = sorted(
#             [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
#             key=lambda i: i["start"],
#         )
#         # res["c_elite"] = Counter(res["elite"])
#         # self.res["token_ids"] = token_ids
#         # res['doclen'] = doclen
#         self.results.doclens.append(doclen)
#         self.results.viz.append([Viz(**i) for i in res['viz']])
# #         self.results.prepare()
#         # self.res.append(res)
#         return doc

#     # getters
#     def has_elite_neg_getter(self, tokens):
#         return any([t._.get("is_elite_neg") for t in tokens])

#     def has_volk_getter(self, tokens):
#         return any([t._.get("is_volk") for t in tokens])

#     def on_hit(self, token, label, span_start, span_end):
#         start = token.idx
#         # end = token.idx + len(token.text) + 1
#         end = token.idx + len(token.text)
#         span_start_idx = span_start.idx
#         span_end_idx = span_end.idx + len(span_end.text)
#         lemma = token._.lemma
#         score = 0.0
#         # label = f"{label} | {score:.2f}"
#         res = {"start": start, "end": end, "coding": label, "score": score, "lemma": lemma, "pos": token.pos_, "dep" : token.dep_, "index": token.i, "ent": token.ent_type_, "sent": token._.sentiws, "idf": self.get_idf(lemma), 'negated': token._.is_negated, "attr_of": token._.attr_of, 'is_elite': token._.is_elite, 'is_elite_neg': token._.is_elite_neg, 'span_start' : span_start_idx, 'span_end' : span_end_idx, 'span_sent': token._.span_sent, 'text': token.text}
#         # res = Viz(**res)
#         return res

#     def get_idf(self, term, idf_weight=1.0):
#         df = self.dictionary.dfs[self.dictionary.token2id[term.lower()]]
#         return tfidfmodel.df2idf(df, self.dictionary.num_docs, log_base=2.0, add=1.0) ** idf_weight

#     @staticmethod
#     def get_viz(token, doclen, label, idf_weight, dictionary=None):
#         start = token.idx
#         end = token.idx + len(token.text) + 1
#         # token = token._.lemma
#         if dictionary:
#             score = ContentAnalysis.compute_score_per_term(token, doclen, idf_weight, dictionary)
#         else:
#             score = 0.0
#         label = f"{label} | {score:.2f}"
#         return {"start": start, "end": end, "coding": label, "lemma": token._.lemma, 'pos': token._.pos_, 'dep' : token._.dep_}


#     @staticmethod
#     def viz(text, row):
#         """visualize documents with displacy"""
#         if isinstance(row, pd.DataFrame):
#             display(row)
#             viz = row.viz[0]
#             ex = [
#                 {
#                     "text": text,
#                     "ents": viz,
#                     "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
#                     # "title": "test",
#                 }
#             ]

#         else:
#             ex = [
#                 {
#                     "text": text,
#                     "ents": viz,
#                     "title": "TEXT",
#                 }
#             ]

#         # find unique labels for coloring options
#         all_ents = {i["label"] for i in viz}
#         options = {"ents": all_ents, "colors": dict()}
#         for ent in all_ents:
#             if ent.startswith("E"):
#                 options["colors"][ent] = "coral"
#             if ent.startswith("V"):
#                 options["colors"][ent] = "lightgrey"
#             if ent.startswith("P"):
#                 options["colors"][ent] = "yellow"

#         displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)


#     @staticmethod
#     def compute_score_per_term(term, doclen, idf_weight, dictionary):
#         score = ContentAnalysis.compute_idf(term, idf_weight, dictionary)
#         ################################
#         res = score / log(doclen+10, 10)
#         ################################
#         return res


#     @staticmethod
#     def compute_idf(term, idf_weight=1.0, dictionary=None):
#         df = dictionary.dfs[dictionary.token2id[term.lower()]]
#         return tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0) ** idf_weight


#     @staticmethod
#     def compute_score_from_counts(counts, doclen, idf_weight, dictionary):
#         scores = []
#         for term, n in counts.items():
#             score = ContentAnalysis.compute_score_per_term(term, doclen, idf_weight, dictionary)
#             scores.append(score * n)
#         res = sum(scores)
#         return res


#     @staticmethod
#     def recount_viz(viz, doclen, dictionary, idf_weight):
#         for i in viz:
#             score = compute_idf(i['lemma'], idf_weight, dictionary)
#             label = i['label']
#             i['label'] = label.replace(label.split('| ')[1], f'{score:.2f}')
#         return viz




# class CustomExtensions(object):

#     name = 'custom_extensions'

#     def __init__(self, nlp):
#         self.lemmatizer = GermaLemma()
#         self.negation_words = set(["nie", "keinsterweise", "keinerweise", "niemals", "nichts", "kaum", "keinesfalls", "ebensowenig", "nicht", "kein", "keine", "weder"])
#         self.negation_cconj = set(['aber', 'jedoch', 'doch', 'sondern'])
#         self.data = dict()
#         self.data['nw'] = list(self.negation_words)
#         self.data['nc'] = list(self.negation_cconj)


#     def __call__(self, doc):
#         Token.set_extension("lemma", getter=self.lemma_getter, force=True)
#         Token.set_extension("is_negation", getter=self.is_negation_getter, force=True)
#         Token.set_extension("is_sentence_break", getter=self.is_sentence_break_getter, force=True)

#         # for token in doc:
#         #     print(token._.lemma, token._.is_negation)

#         return doc

#     def to_disk(self, path, **kwargs):
#         # This will receive the directory path + /my_component
#         path.mkdir(parents=True, exist_ok=True)
#         data_path = path / "data.json"
#         with data_path.open("w", encoding="utf8") as f:
#             f.write(json.dumps(self.data))

#     def from_disk(self, path, **cfg):
#         # This will receive the directory path + /my_component
#         data_path = path / "data.json"
#         with data_path.open("r", encoding="utf8") as f:
#             self.data = json.loads(f)
#         self.negation_words = self.data['nw']
#         self.negation_cconj = self.data['nc']
#         return self


#     def lemma_getter(self, token):
#         # if " " in token.text:
#         #     return token.lemma_.lower()
#         try:
#             return self.lemmatizer.find_lemma(token.text, token.tag_).lower()
#         except:
#             return token.lemma_.lower()

#     def is_negation_getter(self, token):
#         if token._.lemma in self.negation_words:
#             return True
#         else:
#             return False

#     def is_sentence_break_getter(self, token):
#         if token._.lemma in self.negation_cconj:
#             return True
#         else:
#             return False



# def compute_score_from_df(df, dictionary, idf_weight=1.0):
#     cols = ['viz', 'volk', 'volk_attr', 'elite', 'elite_attr']
#     for col in cols:
#         df[col] = df.apply(lambda row: eval(str(row[col])), axis=1)
#     for col in cols[1:]:
#         df[f'c_{col}'] = df.apply(lambda row: Counter(row[col]), axis=1)
#         df[f'score_{col}'] = df.apply(lambda row: ContentAnalysis.compute_score_from_counts(row[f'c_{col}'], row['doclen'], idf_weight, dictionary), axis=1)
#     df['score'] = df.apply(lambda row: sum([row[f'score_{col}'] for col in cols[1:]]), axis=1)


# def evaluate_by_category(category, target, df):
#     grouped = df.groupby(category).mean().sort_values(target, ascending=False)
#     mdbs_meta = pd.read_csv('data/mdbs_metadata.csv')
#     res = pd.merge(grouped, mdbs_meta, how='left', on=category)
#     display(res)

