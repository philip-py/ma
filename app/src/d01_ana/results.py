import copy
from collections import Counter
from datetime import datetime
from math import fabs, log
from pathlib import Path
from typing import List, Optional, Set, Union

import pandas as pd
from app.src.d00_utils.helper import (filter_spans_overlap)
from logzero import setup_logger
from pydantic import BaseModel
from spacy.lang.de.stop_words import STOP_WORDS
from spacy import displacy

def gendocs(label):
    with open("data/corpus_clean/{}".format(label), "r") as text_file:
        return text_file.read()

class Viz(BaseModel):
    start: int
    end: int
    coding: str
    score: float = 0.0
    lemma: str
    pos: str
    dep: str
    index: int
    ent: str
    sent: Optional[float] = 0.0
    idf: float
    negated: bool = False
    is_elite: bool = False
    is_elite_neg: bool = False
    attr_of: Optional[int] = None
    span_start: int
    span_end: int
    span_sent: float = 0.0
    text: str
    E: bool = False
    V: bool = False
    GER: bool = True
    RSN: Set[str] = set()
    TOK_IS_POP: bool = False
    SPAN_IS_POP: bool = False


class Results:
    """Saves results of content analysis and contains mehtods for analysis & visualization"""

    def __init__(self):
        self.vocab = dict()
        # id2token = {value : key for (key, value) in a_dictionary.items()}
        self.labels = []
        self.viz = []
        self.doclens = []
        self.scores = []
        self.counts = []
        self.entities = set()
        self.meta_mdb = None
        self.meta_plenar = None
        self.df = None
        self.spans = []
        self.spans_dict = {}
        self.date = datetime.now()

    def __repr__(self):
        # return 'Results of Content Analysis'
        # return '<{0}.{1} object at {2}>'.format(
        # self.__module__, type(self).__name__, hex(id(self)))
        return f"Results of Analysis on {self.date.strftime('%A, %d %B, %Y')} with {len(self)} Documents."

    def __len__(self):
        return len(self.viz)

    def set_entities(self):
        for doc in self.viz:
            for hit in doc:
                if hit.ent == '':
                    hit.ent = 'ATTR'
                self.entities.add(hit.ent)

    def load_meta(self):
        # self.meta_mdb = pd.read_csv('data/mdbs_metadata.csv')
        self.meta_plenar = pd.read_json(
            'data/plenar_meta.json', orient='index')

    # score is computed here, not in functins above in CA
    def compute_score(self, idf_weight=2.0, sentiment_weight=1.0, doclen_log=10, doclen_min=750, by_doclen=True, post=False):
        scores = []
        labels = ['E', 'EA', 'V', 'VA']
        counts = []
        # seen = set()
        for i, doc in enumerate(self.viz):
            seen = set()
            score_dict = {'score': 0.0}
            count_dict = {}
            for ent in self.entities:
                score_dict[ent] = 0.0
            for label in labels:
                score_dict[label] = 0.0
                count_dict[label] = Counter()
            for hit in doc:
                if not hit.sent:
                    hit.sent = 0.0
                if post:
                    if hit.TOK_IS_POP and hit.SPAN_IS_POP and hit.start not in seen:
                        # score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight) / log(self.doclens[i]+doclen_weight, 10)
                        score = (hit.idf ** idf_weight) * \
                            ((1+fabs(hit.sent)) ** sentiment_weight)
                        seen.add(hit.start)
                    else:
                        score = 0.0
                else:
                    if hit.start not in seen:
                        # score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight) / log(self.doclens[i]+doclen_weight, 10)
                        score = (hit.idf ** idf_weight) * \
                            ((1+fabs(hit.sent)) ** sentiment_weight)
                        seen.add(hit.start)
                if by_doclen:
                    # ln oor log?
                    score = score / log(self.doclens[i] + doclen_min)
                    # score = score / log(self.doclens[i] + doclen_min, doclen_log)
                hit.score = score
                score_dict['score'] += score
                score_dict[hit.ent] += score
                score_dict[hit.coding] += score
                count_dict[hit.coding].update(hit.lemma)
            # for label in labels:
            #     count_dict[label] = Counter(count_dict[label])
            counts.append(count_dict)
            scores.append(score_dict)
        self.scores = scores
        self.counts = counts

    def compute_score_spans(self, by_doclen=True, idf_weight=1.5, sentiment_weight=1.5):
        span_dict = {}
        self.compute_score(
            by_doclen=by_doclen, idf_weight=idf_weight, sentiment_weight=sentiment_weight)
        # {doc: [(span_start, span_end, score_sum)]}
        for i, doc in enumerate(self.viz):
            label = self.labels[i]
            span_dict[label] = {}
            seen = set()
            # scores = []
            for hit in doc:
                if hit.TOK_IS_POP and hit.SPAN_IS_POP and hit.start not in seen:
                    span_start = hit.span_start
                    span_end = hit.span_end
                    span_id = (span_start, span_end)
                    if span_id not in span_dict[label] and span_end not in [i[1] for i in span_dict[label].keys()]:
                        span_dict[label][span_id] = 0.0
                    if span_id in span_dict[label]:
                        span_dict[label][span_id] += hit.score
                    # if span_id not in span_dict[label]:
                    #     span_dict[label][span_id] = 0.0
                    # span_dict[label][span_id] += hit.score
                    seen.add(hit.start)

            # span_dict[label] = sorted(
            #     [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
            #     key=lambda i: i["start"],
            # )

        self.compute_score(by_doclen=True, idf_weight=idf_weight,
                           sentiment_weight=sentiment_weight)
        self.spans_dict = span_dict

    def create_spans(self):
        span_dict = {}
        # {doc: [(span_start, span_end, score_sum)]}
        for i, doc in enumerate(self.viz):
            label = self.labels[i]
            span_dict[label] = {}
            seen = set()
            # scores = []
            for hit in doc:
                hit = hit.dict()
                if hit['start'] not in seen:
                    span_start = hit['span_start']
                    span_end = hit['span_end']
                    span_id = (span_start, span_end)
                    if span_id not in span_dict[label] and span_end not in [i[1] for i in span_dict[label].keys()]:
                        span_dict[label][span_id] = 0.0
                    if span_id in span_dict[label] and span_end not in [i[1] for i in span_dict[label].keys()]:
                        span_dict[label][span_id] += hit['score']
                    seen.add(hit['start'])

            # span_dict[label] = sorted(
            #     [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
            #     key=lambda i: i["start"],
            # )

        # self.compute_score(by_doclen=True, idf_weight=idf_weight, sentiment_weight=sentiment_weight)
        self.spans_dict = span_dict

    def top_spans(self, topn=10):
        all_spans = []
        for doc in self.spans_dict.items():
            for span in doc[1]:
                all_spans.append((doc[0], span, self.spans_dict[doc[0]][span]))
        all_spans.sort(key=lambda tup: tup[2], reverse=True)
        return all_spans[:topn]

    def create_df(self):
        # df = pd.DataFrame.from_dict({'doc': self.labels}, {'doclen': self.doclens}, {'scores': self.scores})
        df = pd.DataFrame.from_dict(
            {'doc': self.labels, 'doclen': self.doclens, 'scores': self.scores})
        df = pd.concat([df.drop('scores', axis=1),
                        df.scores.apply(pd.Series)], axis=1, sort=False)
        self.df = df

    def prepare(self, post=False):
        self.set_entities()
        self.create_spans()
        self.coding_pop()
        self.compute_score(by_doclen=False, idf_weight=1.5,
                           doclen_log=10, post=post)
        self.compute_score_spans()
        # self.create_spans()
        self.create_df()

    def render_online(text, row=None, viz=None, span=None, filter_by=['score'], pres=False):
        """visualize documents with displacy"""

        def filter_by_condition(viz, condition):
            viz = [i for i in viz if i[condition]]
            return viz

        viz = Results.filter_viz(viz, on='start')
        viz = filter_spans_overlap(viz)
        viz_span = []

        if span:
            span = span
        else:
            span = (0, len(text) + 1)

        if pres:
            viz_span_ = []
            for hit in viz:
                paragraph = {}
                hit['start'] -= span[0]
                hit['end'] -= span[0]
                paragraph['start'] = hit['span_start']
                paragraph['end'] = hit['span_end']
                # hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
                if paragraph['start'] not in [i['start'] for i in viz_span_]:
                    viz_span_.append(paragraph)

            for n, v in enumerate(viz_span_):
                viz_span.append({'start': v['start'], 'end': v['end'], 'label': f'P|{n+1}'})

            viz_span = sorted(viz_span, key=lambda x: x['start'])

        ##################################################
        else:

            if filter_by:
                for condition in filter_by:
                    viz = filter_by_condition(viz, condition)

            if span[0] > 0:
                viz = [i for i in viz if i['span_start'] == span[0]]

            for hit in viz:

                hit['start'] -= span[0]
                hit['end'] -= span[0]

                hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
                viz_span.append(hit)

            viz_starts = set([i['span_start'] for i in viz])

            for n, start in enumerate(sorted(viz_starts)):
                if start > 0 and span[0] == 0:
                    viz_span.append({'start': start-1, 'end': start, 'label': f'P{n+1} | {start}'})

            viz_span = sorted(viz_span, key=lambda x: x['start'])
        ###############################################

        ex = [
            {
                "text": text[span[0]: span[1]],
                "ents": viz_span,
                "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
                # 'title': 'text'
            }
        ]
        all_ents = {i["label"] for i in viz_span}

        # else:
        #     viz_all = []
        #     for hit in viz:
        #         hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
        #         viz_all.append(hit)
        #     ex = [
        #         {
        #             "text": text,
        #             "ents": viz_all,
        #             "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
        #         }
        #     ]
        #     # find unique labels for coloring options
        #     all_ents = {i["label"] for i in viz_all}

        options = {"ents": all_ents, "colors": dict()}
        for ent in all_ents:
            if ent.startswith("E"):
                options["colors"][ent] = "coral"
            if ent.startswith("V"):
                options["colors"][ent] = "lightgrey"
            if ent.startswith("P"):
                options["colors"][ent] = "yellow"

        displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)


    def visualize_jupyter(self, label, span=None, filter_by=False, pres=False, online=False):
        if " " in label:
            row = None
            text = label
            viz = copy.deepcopy(self.viz[-1])
        else:
            row = self.df.loc[self.df['doc'] == label]
            text = gendocs(label)
            viz = copy.deepcopy(self.viz[self.labels.index(label)])
        ex, options = Results.render(text, row, viz, span=span,
                       filter_by=filter_by, pres=pres, online=online)
        displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)

    @staticmethod
    def render(text, row, viz, span=None, filter_by=['score'], pres=False, online=False):
        """visualize documents with displacy"""

        def filter_by_condition(viz, condition):
            viz = [i for i in viz if i[condition]]
            return viz

        viz = [i.dict() for i in viz]
        viz = Results.filter_viz(viz, on='start')
        viz = filter_spans_overlap(viz)
        viz_span = []

        # if span:
            # span = span
        # else:
        span = (0, len(text) + 1)

        print(span)

        if pres:
            viz_span_ = []
            for hit in viz:
                paragraph = {}
                # print(span)
                # hit['start'] -= span[0]
                # hit['end'] -= span[0]
                paragraph['start'] = hit['span_start']
                paragraph['end'] = hit['span_end']
                # hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
                if paragraph['start'] not in [i['start'] for i in viz_span_]:
                    viz_span_.append(paragraph)

            for n, v in enumerate(viz_span_):
                viz_span.append(
                    {'start': v['start'], 'end': v['end'], 'label': f'P|{n+1}'})

            viz_span = sorted(viz_span, key=lambda x: x['start'])

        ##################################################
        else:

            if filter_by:
                for condition in filter_by:
                    viz = filter_by_condition(viz, condition)

            if span[0] > 0:
                viz = [i for i in viz if i['span_start'] == span[0]]

            for hit in viz:

                hit['start'] -= span[0]
                hit['end'] -= span[0]

                hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
                viz_span.append(hit)

            viz_starts = set([i['span_start'] for i in viz])

            for n, start in enumerate(sorted(viz_starts)):
                if start > 0 and span[0] == 0:
                    viz_span.append(
                        {'start': start-1, 'end': start, 'label': f'P{n+1} | {start}'})

            viz_span = sorted(viz_span, key=lambda x: x['start'])
        ###############################################

        if online:
            ex = [
                {
                    "text": text[span[0]: span[1]],
                    "ents": viz_span,
                    "title": 'user-input analysis'
                    # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
                    # 'title': 'text'
                }
            ]

        else:
            ex = [
                {
                    "text": text[span[0]: span[1]],
                    "ents": viz_span,
                    "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                    # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
                    # 'title': 'text'
                }
            ]
        all_ents = {i["label"] for i in viz_span}

        options = {"ents": all_ents, "colors": dict()}
        for ent in all_ents:
            if ent.startswith("E"):
                options["colors"][ent] = "coral"
            if ent.startswith("V"):
                options["colors"][ent] = "lightgrey"
            if ent.startswith("P"):
                options["colors"][ent] = "yellow"

        return (ex, options)

    def add_meta_plenar(self):
        df = pd.read_json("data/plenar_meta.json", orient="index")
        df["doc"] = df.index
        df["doc"] = df.doc.apply(lambda x: x.split(".")[0])
        # fix timestamps
        df["date"] = df.datum
        df["date"] = pd.to_datetime(df["date"], unit="ms", errors="ignore")
        # merge results and meta
        dfs = self.df.merge(df.loc[:, ["date", "party", "doc", "name_res", "gender",
                                       "election_list", "education", "birth_year"]], how="left", on="doc")
        dfs = dfs.set_index("date").loc["2013-10-01":"2020-01-01"]
        dfs["date"] = dfs.index
        self.df = dfs

    def evaluate_by_category(self, category, target):
        grouped = self.df.groupby(category).mean(
        ).sort_values(target, ascending=False)
        # mdbs_meta = pd.read_csv('data/mdbs_metadata.csv')
        # res = pd.merge(grouped, mdbs_meta, how='left', on=category)
        return grouped

    def top_terms(self, cat=False, abs=True, party=None, topn=100):
        if party:
            df = self.df.loc[self.df.party == party].copy()
        else:
            df = self.df.copy()
        if abs:
            labels = [i for i in df.doc]
            ids = []
            for label in labels:
                ids.append(self.labels.index(label))
            res = []
            for i, count in enumerate(self.counts):
                if i in ids:
                    if cat:
                        res.append(count[cat])
                    else:
                        res.extend(count.values())
            # res = df.apply(lambda row: Counter(row.counts[cat]), axis=1)
            res = sum([i for i in res], Counter())
        else:
            labels = [i for i in df.doc]
            ids = []
            for label in labels:
                ids.append(self.labels.index(label))

            score_dict = {}

            for i, doc in enumerate(self.viz):
                if i in ids:
                    for hit in doc:
                        if cat:
                            if hit['coding'] == cat:
                                if hit['lemma'] not in score_dict:
                                    score_dict[hit['lemma']] = 0.0
                                score_dict[hit['lemma']] += hit['score']
                        else:
                            if hit['lemma'] not in score_dict:
                                score_dict[hit['lemma']] = 0.0
                            score_dict[hit['lemma']] += hit['score']
            res = score_dict

        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True)[:topn])

    def coded(self, label, index_start, categories=None):
        for hit in self.viz[self.labels.index(label)]:
            # if hit['lemma'] == 'steuerzahler':
            if hit['span_start'] == index_start:
                if not categories:
                    hits.apend(hit)
                else:
                    return({cat: hit[cat] for cat in categories})
        return(hits)

    def coding(self):
        res_viz = []
        # spans = [i for i in self.spans_dict.values()][2].keys()
        spans = [i.keys() for i in self.spans_dict.values()]
        for i, (doc, doc_vizs) in enumerate(zip(list(spans), self.viz)):
            # if i % 500 == 0:
            doc_viz = []
            # doc_vizs = Results.filter_viz(doc_vizs, on='start')
#             for span in self.spans[doc]:

            # self.spans is  not a dict anymore!
            for span in spans[i]:
                viz = []
                # text = gendocs(doc)[span[0]:span[1]]
                viz.extend([viz for viz in doc_vizs if viz.span_start
                            == span[0] and viz.span_end == span[1]])

                # final coding
                pop_hits_v = 0
                pop_hits_e = 0
                for v in viz:
                    v.TOK_IS_POP = False
                    v.SPAN_IS_POP = False

                    if v.GER and (v.V == True or v.E == True):
                        v.TOK_IS_POP = True
                    if v.TOK_IS_POP and v.coding == 'V':
                        pop_hits_v += 1
                        for attr in viz:
                            if attr.attr_of == v.start:
                                attr.V = True
                                attr.TOK_IS_POP = True
                    if v.TOK_IS_POP and (v.coding == 'E' or (v.coding == 'EA' and v.pos == 'NOUN')):
                        pop_hits_e += 1
                        for attr in viz:
                            if attr.attr_of == v.start:
                                attr.E = True
                                attr.TOK_IS_POP = True

                if pop_hits_v > 0 and pop_hits_e > 0:
                    for v in viz:
                        v.SPAN_IS_POP = True
                doc_viz.extend(viz)
            res_viz.append(doc_viz)
        self.viz = res_viz

    def coding_pop(self, idf_weight=1.5, sentiment_weight=1.5):
        self.set_entities()
        self.coding()
        doclen_mean = sum(self.doclens)/len(self.doclens)
        self.compute_score(by_doclen=True, idf_weight=idf_weight,
                           sentiment_weight=sentiment_weight, doclen_log=2, doclen_min=doclen_mean, post=True)
        # self.create_df()
        # self.add_meta_plenar()

    def filter_res(self, label):
        res = Results()
        id = self.labels.index(label)
        res.viz = [self.viz[id]]
        res.labels = [self.labels[id]]
        res.doclens = [self.doclens[id]]
        res.scores = [self.scores[id]]
        res.spans = {label: self.spans[label]}
        return res

    @staticmethod
    def filter_viz(viz, on='start'):
        res = []
        ids = set()
        for hit in viz:
            if hit[on] not in ids:
                res.append(hit)
                ids.add(hit[on])

        return res
