import os
import json
import pickle
from collections import Counter
from datetime import datetime
from math import fabs, log
from pathlib import Path
from typing import List, Optional, Set, Union

import pandas as pd
from app import config
from app.src.d00_utils.helper import(filter_spans_overlap_no_merge)
from germalemma import GermaLemma
from logzero import setup_logger
from spacy import displacy
from spacy.lang.de.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from gensim.models import tfidfmodel

from .results import Results, Viz

class Start(object):
    """Initialize CA"""
    name = "start"

    def __init__(self, nlp):
        self.nlp = nlp
        self.results = Results()

    def __call__(self, doc):
        # calculate doclength
        doclen = len(doc)
        self.results.doclens.append(doclen)
        return doc


class CustomExtensions(object):

    name = 'custom_extensions'

    def __init__(self, nlp, doc_labels):
        self.nlp = nlp
        self.results = self.nlp.pipeline[-1][1].results
        self.lemmatizer = GermaLemma()
        self.negation_words = set(["nie", "keinsterweise", "keinerweise", "niemals", "nichts", "kaum", "keinesfalls", "ebensowenig", "nicht", "kein", "keine", "weder"])
        self.negation_cconj = set(['aber', 'jedoch', 'doch', 'sondern'])
        self.data = dict()
        self.data['nw'] = list(self.negation_words)
        self.data['nc'] = list(self.negation_cconj)
        self.doc_labels = doc_labels
        self.index = 0


    def __call__(self, doc):
        Token.set_extension("lemma", getter=self.lemma_getter, force=True)
        Token.set_extension("is_negation", getter=self.is_negation_getter, force=True)
        Token.set_extension("is_sentence_break", getter=self.is_sentence_break_getter, force=True)

        doc.user_data = {'label': self.doc_labels[self.index]}

        if os.environ.get('FLASK_CONFIG') == 'testing':
            res = {
                "viz": [],
            }
            for token in doc:
                if token._.is_negation:
                    res['viz'].append(self.on_hit(token))
                        # if token._.is_attr and token.i not in token_ids:
                        #     res["elite_attr"].append(token._.lemma)
                        #     res['viz'].append(self.on_hit(token, 'EA', doc[span.start], doc[span.end-1]))
                        #     token_ids.add((token.i, "EA"))
            self.results.viz.append([i for i in res['viz']])

        self.index += 1
        return doc

    def on_hit(self, token):
        start = token.idx
        # end = token.idx + len(token.text) + 1
        end = token.idx + len(token.text)
        lemma = token._.lemma
        res = {"start": start, "end": end, "lemma": lemma, "pos": token.pos_, "dep" : token.dep_, "index": token.i, 'negation': token._.is_negation, 'text': token.text}
        return res

    def to_disk(self, path, **kwargs):
        # This will receive the directory path + /my_component
        path.mkdir(parents=True, exist_ok=True)
        data_path = path / "data.json"
        with data_path.open("w", encoding="utf8") as f:
            f.write(json.dumps(self.data))

    def from_disk(self, path, **cfg):
        # This will receive the directory path + /my_component
        data_path = path / "data.json"
        with data_path.open("r", encoding="utf8") as f:
            self.data = json.loads(f)
        self.negation_words = self.data['nw']
        self.negation_cconj = self.data['nc']
        return self


    def lemma_getter(self, token):
        # if " " in token.text:
        #     return token.lemma_.lower()
        try:
            return self.lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    def is_negation_getter(self, token):
        if token._.lemma in self.negation_words:
            return True
        else:
            return False

    def is_sentence_break_getter(self, token):
        if token._.lemma in self.negation_cconj:
            return True
        else:
            return False



class SentimentRecognizer(object):

    name = "sentiment_recognizer"

    def __init__(self, nlp):
        self.nlp = nlp
        self.results = Results()
        self.load_dicts()
        # Token.set_extension('is_neg', default=False, force=True)
        # Token.set_extension('is_pos', default=False, force=True)
        Token.set_extension("is_neg", getter=self.is_neg_getter, force=True)
        Token.set_extension("is_pos", getter=self.is_pos_getter, force=True)
        Token.set_extension("is_negated", getter=self.is_negated_getter, force=True)
        Token.set_extension("span_sent", default=None, force=True)
        Doc.set_extension("has_neg", getter=self.has_neg, force=True)
        Doc.set_extension("has_pos", getter=self.has_pos, force=True)
        Span.set_extension("has_neg", getter=self.has_neg, force=True)
        Span.set_extension("has_pos", getter=self.has_pos, force=True)

    def __call__(self, doc):
        return doc

    def is_neg_getter(self, token):
        if token._.lemma in self.negativ:
            if token._.is_negated:
                return False
            else:
                return True
        if token._.lemma in self.positiv:
            if token._.is_negated:
                return True
            else:
                return False

    def is_pos_getter(self, token):
        if token._.lemma in self.positiv:
            if token._.is_negated:
                return False
            else:
                return True
        if token._.lemma in self.negativ:
            if token._.is_negated:
                return True
            else:
                return False

    def is_negated_getter(self, token):

        check = list(token.children)
        node = token
        # CAREFUL HERE
        if token.pos_ == "ADJ" or token.pos_ == "ADV":
            if token.i - 1 >= 0:
                check.append(token.doc[token.i - 1])
        ####################
        while node.head:
            seen = node
            if seen == node.head:
                # CAREFUL HERE
                check.extend(list(node.head.lefts))
                break
            check.append(node)
            check.extend(list(node.children))
            if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
                check.append(node.head)
                # CAREFUL HERE #####
                if node.head.dep_ == 'root':
                    check.exten(list(node.head.lefts))
                ####################
                break
            else:
                node = node.head
        attr = [
            # child for child in check if child.dep_ == "ng" or child._.lemma in negation_words
            child for child in check if child.dep_ == "ng" or child._.is_negation
        ]
        if attr:
            return True
        else:
            return False

    def load_dicts(self):
        # dict_folder = "dict"
        dict_folder = os.path.join(config[os.getenv('FLASK_CONFIG')].DIR_DATA, 'dict')
        sent = pd.read_csv(f"{dict_folder}/SentDict.csv")
        self.positiv = set([
                x.strip()
                for x in sent.loc[sent.sentiment == 1, ["feature"]]["feature"].tolist()
        ])
        self.negativ = set([
                x.strip()
                for x in sent.loc[sent.sentiment == -1, ["feature"]]["feature"].tolist()
        ])

    def has_neg(self, tokens):
        return any([t._.get("is_neg") for t in tokens])

    def has_pos(self, tokens):
        return any([t._.get("is_pos") for t in tokens])

class EntityRecognizer(object):

    name = "entity_recognizer"

    def __init__(self, nlp, config_ca):
        self.nlp = nlp
        self.results = Results()
        self.load_dicts()
        self.ruler = EntityRuler(nlp, overwrite_ents=True, phrase_matcher_attr="LOWER")
        self.vocab = nlp.vocab
        self.config = config_ca
        self.debug = False
        if self.config.debug:
            self.debug = True

        
        patterns = []
        for term in self.dict_people:
            patterns.append({"label": "PEOPLE", "pattern": [{"_": {"lemma": term}}]})
        for term in self.dict_elite:
            patterns.append({"label": "ELITE", "pattern": [{"_": {"lemma": term}}]})
        for term in self.dict_elite_standalone:
            patterns.append(
                {"label": "ELITE_STANDALONE", "pattern": [{"_": {"lemma": term}}]}
            )
        for term in self.dict_people_ord:
            patterns.append(
                {"label": "PEOPLE_ORD", "pattern": [{"_": {"lemma": term}}]}
            )
        for term in self.dict_people_ger:
            patterns.append(
                {"label": "PEOPLE_GER", "pattern": [{"_": {"lemma": term}}]}
            )
        for term in self.dict_attr_ord:
            patterns.append({"label": "ATTR_ORD", "pattern": [{"_": {"lemma": term}}]})
        for term in self.dict_attr_ger:
            patterns.append({"label": "ATTR_GER", "pattern": [{"_": {"lemma": term}}]})
        self.ruler.add_patterns(patterns)
        # self.ruler.add_patterns([{'label': 'ELITE', 'pattern': 'europäische union'}])

        Token.set_extension("is_volk", default=False, force=True)
        Token.set_extension("is_elite", default=False, force=True)
        Token.set_extension("is_elite_neg", default=False, force=True)
        Token.set_extension("is_attr", default=False, force=True)
        Token.set_extension("attr_of", default=None, force=True)
        Doc.set_extension("has_volk", getter=self.has_volk, force=True)
        Doc.set_extension("has_elite", getter=self.has_elite, force=True)
        Span.set_extension("has_volk", getter=self.has_volk, force=True)
        Span.set_extension("has_elite", getter=self.has_elite, force=True)

    def __call__(self, doc):

        matches = self.ruler.matcher(doc)
        # matches.extend(self.ruler.phrase_matcher(doc))
        spans = []
        for id, start, end in matches:
            entity = Span(doc, start, end, label=self.vocab.strings[id])
            spans.append(entity)
        filtered = filter_spans(spans)
        for entity in filtered:
            # People setter
            if entity.label_ == "PEOPLE":
                for token in entity:
                    token._.set("is_volk", True)
            if entity.label_ == "PEOPLE_ORD":
                for token in entity:
                    check = list(token.children)
                    attr = set(
                        [
                            child
                            for child in check
                            if child._.lemma.lower() in self.dict_attr_ord
                        ]
                    )
                    if attr:
                        token._.set("is_volk", True)
                        for child in attr:
                            child._.set("is_volk", True)
                            child._.set("is_attr", True)
                            child._.set("attr_of", token.idx)

            if entity.label_ == "PEOPLE_GER" or entity.label_ == "PEOPLE_ORD":
                for token in entity:
                    check = list(token.children)
                    attr = set(
                        [
                            child
                            for child in check
                            if child._.lemma.lower() in self.dict_attr_ger
                        ]
                    )
                    if attr:
                        token._.set("is_volk", True)
                        for child in attr:
                            child._.set("is_volk", True)
                            child._.set("is_attr", True)
                            child._.set("attr_of", token.idx)
            # Elite setter
            if entity.label_ == "ELITE":
                for token in entity:
                    token._.set("is_elite", True)

                    check = list(token.children)
                    node = token
                    while node.head:
                        seen = node
                        for t in node.children:
                            if t.dep_ == "conj":
                                break
                            check.append(t)
                            # for tok in t.children:
                            # #     check.append(tok)
                            #     if tok.dep_ == "pd":
                            #         check.append(tok)
                            #     elif tok.dep_ == "mo":
                            #         check.append(tok)
                            #     elif tok.dep_ == "oa":
                            #         check.append(tok)
                            #     elif tok.dep_ == "oa2":
                            #         check.append(tok)
                            #     elif tok.dep_ == "og":
                            #         check.append(tok)
                            #     elif tok.dep_ == "da":
                            #         check.append(tok)
                            #     elif tok.dep_ == "op":
                            #         check.append(tok)
                            #     elif tok.dep_ == "cc":
                            #         check.append(tok)
                            #     elif tok.dep_ == 'avc':
                            #         check.append(tok)
                            #     elif tok.dep_ == 'app':
                            #         check.append(tok)
                            #     elif tok.dep_ == 'adc':
                            #         check.append(tok)
                            #     elif tok.dep_ == 'ag':
                            #         check.append(tok)
                        check.append(node)
                        # check.extend(list(node.children))
                        if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
                            check.append(node.head)
                            break
                        # if node.head.pos_ == 'CCONJ' and node.head.text in negation_cconj:
                        if node.head.pos_ == 'CCONJ' and node.head._.is_sentence_break:
                            check.append(node.head)
                            break
                        if seen == node.head:
                            break
                        else:
                            node = node.head
                    attr = set([child for child in check if child._.is_neg])
                    if attr:
                        token._.set("is_elite_neg", True)
                        for child in attr:
                            child._.set("is_elite_neg", True)
                            child._.set("is_attr", True)
                            child._.set("attr_of", token.idx)

            # if entity.label_ == "ELITE" or entity.label_ == "ELITE_STANDALONE":
            if entity.label_ == "ELITE_STANDALONE":
                for token in entity:
                    token._.set("is_elite", True)
                    if not token._.is_negated:
                        token._.set("is_elite_neg", True)
            doc.ents = list(doc.ents) + [entity]
            
        if self.debug:
            print(doc.ents)
        # nach content analyse?
        # for span in filtered:
        # span.merge()
        return doc

    def load_dicts(self):
        # dict_folder = "dict"
        dict_folder = os.path.join(config[os.getenv('FLASK_CONFIG')].DIR_DATA, 'dict')
        # import all dicts
        # elite
        df_elite = pd.read_csv(f"{dict_folder}/elite_dict.csv")
        self.dict_elite = set(
            df_elite[df_elite.type != "elite_noneg"]["feature"].tolist()
        )
        self.dict_elite_standalone = set(
            df_elite[df_elite.type == "elite_noneg"]["feature"].tolist()
        )

        # people
        df_people = pd.read_csv(f"{dict_folder}/people_dict.csv")
        self.dict_people = set(
            df_people[df_people.type == "people"]["feature"].tolist()
        )
        self.dict_people_ord = set(
            df_people[df_people.type == "people_ordinary"]["feature"].tolist()
        )
        self.dict_attr_ord = set(
            df_people[df_people.type == "attr_ordinary"]["feature"].tolist()
        )
        self.dict_people_ger = set(
            df_people[df_people.type == "people_ger"]["feature"].tolist()
        )
        self.dict_attr_ger = set(
            df_people[df_people.type == "attr_ger"]["feature"].tolist()
        )

        # testing:
        # self.dict_people.add("wir sind das volk")
        # self.dict_elite.add("europäische union")


    # getters
    def has_volk(self, tokens):
        return any([t._.get("is_volk") for t in tokens])

    def has_elite(self, tokens):
        return any([t._.get("is_elite") for t in tokens])


class ContentAnalysis(object):
    """Runs Content Analysis as spacy-pipeline-component"""
    name = "content_analysis"

    def __init__(self, nlp, config_ca, window_size=25):
        self.nlp = nlp
        self.config = config_ca
        self.debug = False
        if self.config.debug:
            self.debug = True
        self.dictionary = pickle.load(open(f"{config[os.getenv('FLASK_CONFIG')].DIR_DATA}/plenar_dict.pkl", "rb"))
        # self.dictionary = None
        # Results()
        # self.res = []
        # self.results = Results()
        self.results = Results()
        self.window_size = window_size


        Span.set_extension(
            "has_elite_neg", getter=self.has_elite_neg_getter, force=True
        )
        Span.set_extension(
            "has_volk", getter=self.has_volk_getter, force=True
        )

    def __call__(self, doc):
        res = {
            "viz": [],
            "volk": [],
            "volk_attr": [],
            "elite": [],
            "elite_attr": [],
        }

        ##########################################
        window_size = self.window_size
        # idf_weight = 1.0
        ##########################################
        
        # matcher cant find shit without is_elite_neg from entity recognizer!
        matcher = Matcher(self.nlp.vocab)
        pattern = [{"_": {"is_elite_neg": True}}]
        matcher.add("text", None, pattern)
        matches = matcher(doc)
        doclen = len(doc)

        # spans = set()
        spans = []
        token_ids = set()
        ps_counter = 1
        last_start = None
        for id, start, end in matches:
            if start - window_size < 0:
                start = 0
            else:
                start = start - window_size
            if end + window_size > doclen:
                end = doclen
            else:
                end = end + window_size
            sentence_start = doc[start].sent.start
            sentence_end = doc[end-1].sent.end
            # span = doc[start - window_size : end + window_size]
            span = {'span_start': sentence_start, 'span_end': sentence_end}
            spans.append(span)

            """keep
            span = doc[sentence_start : sentence_end]
            spans.add(span)
            """

        # CAREFUL!!!!!
        spans = filter_spans_overlap_no_merge(spans)
#         print(spans)
        for span in spans:
            span = doc[span['span_start'] : span['span_end']]
            if span._.has_elite_neg and span._.has_volk:
                # check sentiment of span mit sentiws
                span_sentiment = sum([token._.sentiws for token in span if token._.sentiws])
                # if span_sentiment > 0.0:
                #     pass
                # else:
                for token in span:
                    token._.span_sent = span_sentiment
                    if token._.is_volk:
                        # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "V", idf_weight, dictionary=self.dictionary))
                        if token._.is_attr and token.i not in token_ids:
                            res["volk_attr"].append(token._.lemma)
                            res['viz'].append(self.on_hit(token, 'VA', doc[span.start], doc[span.end-1]))
                            token_ids.add((token.i, "VA"))
                        else:
                            if token.i not in token_ids:
                                res["volk"].append(token._.lemma)
                                res['viz'].append(self.on_hit(token, 'V', doc[span.start], doc[span.end-1]))
                                token_ids.add((token.i, "V"))

                    if token._.is_elite_neg:
                        # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "E", idf_weight, dictionary=self.dictionary))
                        if token._.is_attr and token.i not in token_ids:
                            res["elite_attr"].append(token._.lemma)
                            res['viz'].append(self.on_hit(token, 'EA', doc[span.start], doc[span.end-1]))
                            token_ids.add((token.i, "EA"))
                        else:
                            if token.i not in token_ids:
                                res["elite"].append(token._.lemma)
                                res['viz'].append(self.on_hit(token, 'E', doc[span.start], doc[span.end-1]))
                                token_ids.add((token.i, "E"))

        # sorts by start AND deletes duplicates!
        res["viz"] = sorted(
            [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
            key=lambda i: i["start"],
        )
        # res["c_elite"] = Counter(res["elite"])
        # self.res["token_ids"] = token_ids
        # res['doclen'] = doclen
        self.results.doclens.append(doclen)
        self.results.viz.append([Viz(**i) for i in res['viz']])
        if self.debug:
            print(self.results.viz)
#         self.results.prepare()
        # self.res.append(res)
        return doc

    # getters
    def has_elite_neg_getter(self, tokens):
        return any([t._.get("is_elite_neg") for t in tokens])

    def has_volk_getter(self, tokens):
        return any([t._.get("is_volk") for t in tokens])

    def on_hit(self, token, label, span_start, span_end):
        start = token.idx
        # end = token.idx + len(token.text) + 1
        end = token.idx + len(token.text)
        span_start_idx = span_start.idx
        span_end_idx = span_end.idx + len(span_end.text)
        lemma = token._.lemma
        score = 0.0
        # label = f"{label} | {score:.2f}"
        res = {"start": start, "end": end, "coding": label, "score": score, "lemma": lemma, "pos": token.pos_, "dep" : token.dep_, "index": token.i, "ent": token.ent_type_, "sent": token._.sentiws, "idf": self.get_idf(lemma), 'negated': token._.is_negated, "attr_of": token._.attr_of, 'is_elite': token._.is_elite, 'is_elite_neg': token._.is_elite_neg, 'span_start' : span_start_idx, 'span_end' : span_end_idx, 'span_sent': token._.span_sent, 'text': token.text}
        # res = Viz(**res)
        return res

    def get_idf(self, term, idf_weight=1.0):
        df = self.dictionary.dfs[self.dictionary.token2id[term.lower()]]
        return tfidfmodel.df2idf(df, self.dictionary.num_docs, log_base=2.0, add=1.0) ** idf_weight

    @staticmethod
    def get_viz(token, doclen, label, idf_weight, dictionary=None):
        start = token.idx
        end = token.idx + len(token.text) + 1
        # token = token._.lemma
        if dictionary:
            score = ContentAnalysis.compute_score_per_term(token, doclen, idf_weight, dictionary)
        else:
            score = 0.0
        label = f"{label} | {score:.2f}"
        return {"start": start, "end": end, "coding": label, "lemma": token._.lemma, 'pos': token._.pos_, 'dep' : token._.dep_}


    @staticmethod
    def viz(text, row):
        """visualize documents with displacy"""
        if isinstance(row, pd.DataFrame):
            display(row)
            viz = row.viz[0]
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                    # "title": "test",
                }
            ]

        else:
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": "TEXT",
                }
            ]

        # find unique labels for coloring options
        all_ents = {i["label"] for i in viz}
        options = {"ents": all_ents, "colors": dict()}
        for ent in all_ents:
            if ent.startswith("E"):
                options["colors"][ent] = "coral"
            if ent.startswith("V"):
                options["colors"][ent] = "lightgrey"
            if ent.startswith("P"):
                options["colors"][ent] = "yellow"

        displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)


    @staticmethod
    def compute_score_per_term(term, doclen, idf_weight, dictionary):
        score = ContentAnalysis.compute_idf(term, idf_weight, dictionary)
        ################################
        res = score / log(doclen+10, 10)
        ################################
        return res


    @staticmethod
    def compute_idf(term, idf_weight=1.0, dictionary=None):
        df = dictionary.dfs[dictionary.token2id[term.lower()]]
        return tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0) ** idf_weight


    @staticmethod
    def compute_score_from_counts(counts, doclen, idf_weight, dictionary):
        scores = []
        for term, n in counts.items():
            score = ContentAnalysis.compute_score_per_term(term, doclen, idf_weight, dictionary)
            scores.append(score * n)
        res = sum(scores)
        return res


    @staticmethod
    def recount_viz(viz, doclen, dictionary, idf_weight):
        for i in viz:
            score = compute_idf(i['lemma'], idf_weight, dictionary)
            label = i['label']
            i['label'] = label.replace(label.split('| ')[1], f'{score:.2f}')
        return viz


class Spans(object):
    """DOCSTRING"""
    name = "spans"

    def __init__(self, nlp):
        self.nlp = nlp
        self.results = self.nlp.pipeline[-1][1].results
        self.results.spans = []
        self.index = 0

    def __call__(self, doc):
        spans = []
        for hit in self.results.viz[self.index]:
            span_start = hit.span_start
            span_end = hit.span_end
            span_id = (span_start, span_end)
            if span_id not in spans:
                spans.append(span_id)
        self.results.spans.append(spans)
        self.index += 1
        return doc


def gendocs(label):
    with open("data/corpus_clean/{}.txt".format(label), "r") as text_file:
        return text_file.read()


