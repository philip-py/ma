# from transformers import pipeline
from pprint import pprint
from app import db
from app.models import Doc as Document

# clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=0)

class Clf(object):

    name = 'clf'

    def __init__(self, nlp, config):
        self.nlp = nlp
        self.config = config
        self.model = config.clf_model
        if self.model:
            self.clf = pipeline("zero-shot-classification", model=self.model, device=0)
        self.results = self.nlp.pipeline[-1][1].results
        self.index = 0

    def __call__(self, doc):
        debug = self.config.debug

        if debug:
            print(doc.user_data.get('label'))

        self.clf_ger(doc, debug=debug)
        self.clf_pop(doc, debug=debug)
        self.clf_demo(doc, debug=debug)
        self.iterate()

        return doc

    def iterate(self):
        self.index +=1

    def get_results(self, n: int = -1):
        # for element in self.nlp.pipeline:
            # if element[0] == 'content_analysis':
                # res = element[1]
        return self.nlp.pipeline[n][1].results

    def clf_ger(self, doc, debug=False):
        doc_vizs = self.results.viz[self.index]
        res_viz = []
        seen_span = set()
        for span in self.results.spans[self.index]:
            viz = []
            span_id = (span[0], span[1])
            text = doc.text[span[0]:span[1]]
            if span_id not in seen_span:
                viz.extend([viz for viz in doc_vizs if viz.span_start == span[0] and viz.span_end == span[1]])
                seen_span.add(span_id)

            # 1. check if text is ger
            hypothesis_template = 'Der Text handelt von {}'
            candidate_labels = ['Deutschland', 'Europa', 'Ausland']
            s = self.clf(text, candidate_labels, hypothesis_template, multi_class=True)
            # if s['labels'][0] == 'Ausland' and s['scores'][0] > 0.5:
            id_ausland = s['labels'].index('Ausland')
            id_ger = s['labels'].index('Deutschland')
            if s['labels'][-1] == 'Deutschland' and s['scores'][id_ausland] > 0.5:
                for v in viz:
                    v.GER = False

            elif s['labels'][0] == 'Ausland' and s['scores'][id_ausland] / s['scores'][id_ger] >  2:
                for v in viz:
                    v.GER = False

            if self.config.debug:
                pprint(span_id)
                pprint(hypothesis_template)
                pprint(s)

            res_viz.extend(viz)
        self.results.viz[self.index] = res_viz
#         self.index += 1

    def _clf_ger(self, doc, debug=False):
        doc_vizs = self.results.viz[self.index]
        res_viz = []
        seen_span = set()
        for span in self.results.spans[self.index]:
            viz = []
            span_id = (span[0], span[1])
            text = doc.text[span[0]:span[1]]
            if span_id not in seen_span:
                viz.extend([viz for viz in doc_vizs if viz.span_start == span[0] and viz.span_end == span[1]])
                seen_span.add(span_id)
            for v in viz:
                v.score = 42.0
            if self.config.debug:
                pprint(span_id)
            res_viz.extend(viz)
        self.results.viz[self.index] = res_viz
        self.index += 1

    def clf_pop(self, doc, debug=False):
        doc_vizs = self.results.viz[self.index]
        res_viz = []
        seen_span = set()
        for span in self.results.spans[self.index]:
            viz = []
            span_id = (span[0], span[1])
            text = doc.text[span[0]:span[1]]
            if span_id not in seen_span:
                viz.extend([viz for viz in doc_vizs if viz.span_start == span[0] and viz.span_end == span[1]])
                seen_span.add(span_id)

                # 2. check if volk is benachteiligt:
                condition = False
                while not condition:
                    h0 = '{} hat Nachteile'
                    # h1 = 'ungerecht für {}'
                    candidate_labels = set()
                    for v in viz:
                        if v.coding == 'V':
                            candidate_labels.add(v.lemma)
                    candidate_labels = list(candidate_labels)
                    hs = [h0]
                    for h, hypothesis_template in enumerate(hs):
                        if hypothesis_template and candidate_labels:
                            s = self.clf(text, candidate_labels, hypothesis_template, multi_class=True)
                        candidates_people = []
                        for j, label in enumerate(s['labels']):
                            if s['scores'][j] >= 0.75:
                                candidates_people.append(label)
                                for v in viz:
                                    if v.lemma == label:
                                        v.V = True
                                        v.RSN.add(h)
                                condition = True
                            if self.config.debug:
                                pprint(hypothesis_template)
                                pprint(s)
                    condition = True

                # 3. check if elite benachteiligt volk:
                for volk in candidates_people:
                    condition = False
                    while not condition:
                        h0 = '{} benachteiligt ' + volk
                        h1 = '{} entmachtet ' + volk
                        h2 = '{} betrügt ' + volk
                        # h3 = '{} belügt ' + volk
                        candidate_labels = set()
                        for v in viz:
                            if v.coding == 'E' or (v.coding == 'EA' and v.pos == 'NOUN'):
                                candidate_labels.add(v.lemma)
                        candidate_labels = list(candidate_labels)
                        hs = [h0, h1, h2]
                        for h, hypothesis_template in enumerate(hs):
                            if candidate_labels:
                                s = self.clf(text, candidate_labels, hypothesis_template, multi_class=True)
                                for j, label in enumerate(s['labels']):
                                    if s['scores'][j] >= 0.75:
                                        for v in viz:
                                            if v.lemma == label:
                                                v.E = True
                                                v.RSN.add(h)
                                        condition=True
                                if self.config.debug:
                                    pprint(hypothesis_template)
                                    pprint(s)
                        condition=True
            res_viz.extend(viz)
        self.results.viz[self.index] = res_viz

    def clf_demo(self, doc, debug=False):
        doc_vizs = self.results.viz[self.index]
        res_viz = []
        seen_span = set()
        for span in self.results.spans[self.index]:
            viz = []
            span_id = (span[0], span[1])
            text = doc.text[span[0]:span[1]]
            checked_history=False
            is_present = True
            if span_id not in seen_span:
                viz.extend([viz for viz in doc_vizs if viz.span_start == span[0] and viz.span_end == span[1]])
                seen_span.add(span_id)
                demo = ['Demokratie', 'Gewaltenteilung', 'Gerechtigkeit', 'Meinungsfreiheit']
                for w in demo:
                    if w in text:
                        if not checked_history:
                            hypothesis_template = 'Der Text beschreibt {}'
                            candidate_labels = ['Geschichte', 'Nationalsozialismus']
                            s = self.clf(text, candidate_labels, hypothesis_template, multi_class=True)
                            if debug:
                                print(s)
                            if any(i > 0.75 for i in s['scores']):
                                is_present=False
                                checked_history=True
                        if is_present:
                            # REASON IS S
                            hypothesis_template = 'In Deutschland herrscht keine {}'
                            candidate_labels = [w]
                            s = self.clf(text, candidate_labels, hypothesis_template, multi_class=True)
                            if s['scores'][0] > 0.75:
                                for v in viz:
                                    if v.coding.startswith('E'):
                                        v.E = True
                                        v.RSN.add('S')
                                    elif v.coding.startswith('V'):
                                        v.V = True
                                        v.RSN.add('S')
                            if self.config.debug:
                                pprint(hypothesis_template)
                                pprint(s)
            res_viz.extend(viz)
        self.results.viz[self.index] = res_viz

# def clf_ger(res, debug=False):
#     res_viz = []
#     for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
#         if i % 5 == 0:
#             print(i, f'/{len(res.spans)}')
#         doc_viz = []
#         seen_span = set()
#         # doc_vizs = Results.filter_viz(doc_vizs, on='start')
#         for span in res.spans[doc]:
#             viz = []
#             span_id = (span[0], span[1])
# #             text = gendocs(doc)[span[0]:span[1]]
#             document = db.session.query(Document).filter_by(id=doc).first()
#             text = document.text
#             # viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['start'] - viz['span_start'] <= 2_400])

#             if span_id not in seen_span:
#                 viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['span_end'] == span[1]])
#                 seen_span.add(span_id)

#             for v in viz:
#                 v['RLY_GER'] = True

#             # if viz:
#             # 1. check if text is ger
#             hypothesis_template = 'Der Text handelt von {}'
#             candidate_labels = ['Deutschland', 'Europa', 'Ausland']
#             s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
#             # if s['labels'][0] == 'Ausland' and s['scores'][0] > 0.5:
#             id_ausland = s['labels'].index('Ausland')
#             id_ger = s['labels'].index('Deutschland')
#             if s['labels'][-1] == 'Deutschland' and s['scores'][id_ausland] > 0.5:
#                 for v in viz:
#                     v['RLY_GER'] = False

#             elif s['labels'][0] == 'Ausland' and s['scores'][id_ausland] / s['scores'][id_ger] >  2:
#                 for v in viz:
#                     v['RLY_GER'] = False

#             ######################################
#             # 1. check if text is ger v2:
#             # hypothesis_template = 'Der Text beschreibt {}'
#             # candidate_labels = ['Deutschland', 'Ausland']
#             # s = clf(text, candidate_labels, hypothesis_template, multi_class=False)
#             # if s['labels'][0] == 'Ausland' and s['scores'][0] >= 0.9:
#             #     for v in viz:
#             #         v['RLY_GER'] = False
#             #####################################

#             if debug:
#                 pprint(span_id)
#                 pprint(hypothesis_template)
#                 pprint(s)

#             doc_viz.extend(viz)
#         res_viz.append(doc_viz)

#     return res_viz
