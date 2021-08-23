from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc
import pandas as pd
# from src.d00_utils.del_chars import del_chars
import re
from pprint import pprint
import string
import pickle
from tqdm import tqdm
from app import create_app, db
from app.models import Doc as Document
from app.models import Akteur
from app.src.d00_utils.helper import (filter_spans_overlap, filter_viz)
from sqlalchemy.sql import text
from app.src.d01_ana.diskurs import MyCorpus, doc_gen
from sqlalchemy import and_
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from random import sample
import pickle
import spacy
from spacy.tokens import Doc, Span, Token

abks = ['a.a.O.',
 'Abb.',
 'Abh.',
 'Abk.',
 'allg.',
 'bes.',
 'bez.',
 'Bez.',
 'bzw.',
 'eigtl.',
 'erg.',
 'geb.',
 'gegr.',
 'Ggs.',
 'i.e.S.',
 'i.w.S.',
 'jmd.',
 'jmdm.',
 'jmdn.',
 'jmds.',
 'o.Ä.',
 'scherzh.',
 'u.',
 'u.a.',
 'u.Ä.',
 'übertr.',
 'u.dgl.',
 'ugs.',
 'urspr.',
 'usw.',
 'zz.',
 'zzt.',
 'm.E.',]

def clean_corpus(text_gen):
    def remove_URL(sample):
        """Remove URLs from a sample string"""
        pattern=r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))"""
        # return re.sub(r"http\S+", "", sample)
        return re.sub(pattern, "", sample)
    # PATH = 'data/corpus'
    # PATH = '/media/philippy/SSD/data/ma/twitter/'
    # files = glob.glob(PATH + '*')

    # df_gen = (pd.read_json(f) for f in files)
    corpus = Corpus()
    # for
    for doc in text_gen:
    # corpus.add_folder(PATH + '/presse')
        corpus.add_doc(str(doc.id), remove_URL(doc.text))
        # corpus.add_doc(str(doc.id), doc.text)

    # corpus = Corpus.from_folder(PATH + '/plenar', encoding='utf8')
    # corpus = Corpus.from_folder(PATH + '/plenar')
    # corpus.add_folder(PATH + '/presse')
    # corpus.add_folder(PATH + '/twitter')

    doc_labels = corpus.get_doc_labels(sort=True)

    table_umlauts = {"ÃŸ": "ß", "ãÿ": "ß", "ã¤": "ä", "ã¼": "ü", "ã¶": "ö", 'Ã„': 'Ä', "Ãœ": "Ü", "Ã–": "Ö", 'â‚¬': '€'}

    table_chars = {';': '.', '$': '', '?': '.', '!': '.', ':':'.', '@': '', '#': ''}
    left = corpus.unique_characters - set(string.printable)
    umlauts = ['ä', 'ü', 'ö', 'Ä', 'Ö', 'Ü', 'ß']
    for um in umlauts:
        left.discard(um)
    for char in left:
        if char not in table_chars:
            table_chars[char] = ''
    keep = ['.', ',']
    for char in string.punctuation:
        if char not in table_chars and char not in keep:
            table_chars[char] = ''

    abk = {k: k.replace('.', '') for k in abks}
    # print(table_chars)

    # phrases = {'teilentweetPrint': '', 'Current Page': '', 'Pressekontakt .   CDUCSU  BundestagsfraktionPressestelleTelefon .   030 22752360Fax .       030 22756660Internet .  http . www . cducsu . deEmail .  pressestellecducsu . de OriginalContent von .  CDUCSU  Bundestagsfraktion, übermittelt durch news aktuell': '', }


    def repl_phrases(doc):
        for k, v in phrases.items():
            doc = doc.replace(k,v)
        return doc

    def repl_abk(doc):
        for k, v in abk.items():
            doc = doc.replace(k,v)
        return doc

    def repl_umlauts(doc):
        for k, v in table_umlauts.items():
            doc = doc.replace(k,v)
        return doc

    def repl_chars(doc):
        for k, v in table_chars.items():
            doc = doc.replace(k, v)
        return doc

    def repl_nl(doc):
        doc = doc.replace(r'\n', "")
        return doc

    def repl_last(doc):
        doc = doc.replace('-', ' ')
        return doc

    def repl_dot(doc):
        doc = doc.replace('.', ' . ')
        return doc

    def fix_spaces(doc):
        doc = ' '.join(doc.split())
        return doc

    corpus.apply(lambda x: repl_umlauts(x))
    corpus.apply(lambda x: repl_chars(x))
    corpus.apply(lambda x: repl_abk(x))
    # corpus.apply(lambda x: repl_nl(x))

    # corpus.replace_characters(del_chars)

    # correct contractions
    pttrn_contraction_ws = re.compile(r'(\w+)(\s+)(-\w+)')
    corpus.apply(lambda t: pttrn_contraction_ws.sub(lambda m: m.group(1) + m.group(3), t))

    def replace_nums(text):
        # nums = re.findall(r'[0-9]+ ?\.( ?[0-9]+)?', text)
        # nums = re.findall(r'[0-9]+\ ?\.( ?[0-9]+)?', text)
        nums = re.findall(r'[0-9]+ ?\.( ?[0-9]+)?', text)
        for match in nums:
            new = match.replace('.', '')
            text = text.replace(match, new)
        # return re.sub(pttrn_dots, ".", sample)
        return text

    corpus.apply(lambda t: replace_nums(t))

    def replace_nums2(text):
        nums = re.findall(r'[0-9]+ ?\.', text)
        for match in nums:
            new = match.replace('.', '')
            text = text.replace(match, new)
        # return re.sub(pttrn_dots, ".", sample)
        return text

    corpus.apply(lambda t: replace_nums2(t))

    corpus.apply(lambda x: repl_last(x))
    corpus.apply(lambda x: repl_dot(x))
    # corpus.apply(lambda x: repl_phrases(x))
    def remove_dots(sample):
        pttrn_dots = re.compile(r'(\. ?)+')
        return re.sub(pttrn_dots, ".", sample)
    corpus.apply(lambda x: remove_dots(x))

    corpus.apply(fix_spaces)

    # delete special chars in tweets:
    # left = corpus.unique_characters - set(string.printable)
    # umlauts = ['ä', 'ü', 'ö', 'Ä', 'Ö', 'Ü', 'ß']
    # for um in umlauts:
    #     left.discard(um)
    # left_dict = {d: None for d in left}

    # corpus.replace_characters(left_dict)


    # for i in range(500):
    #     print(corpus[str(i+1)])

    print('these non-ASCII characters are left:')
    pprint(corpus.unique_characters - set(string.printable))

    # print(table_chars)
    for label in doc_labels:
        yield(corpus[str(label)])

def clean_name_res(dfp):
    dfp = dfp.rename(columns={"name_res": "speaker_cleaned"})

    dfp.loc[dfp['speaker_cleaned'] == 'Angela Merkel,',
            'speaker_cleaned'] = 'Angela Merkel'
    dfp.loc[dfp['speaker_cleaned'] == 'Dagmar G. Wöhrl',
            'speaker_cleaned'] = 'Dagmar Wöhrl'
    dfp.loc[dfp['speaker_cleaned'] == 'Andreas G. Lämmel',
            'speaker_cleaned'] = 'Andreas Lämmel'
    dfp.loc[dfp['speaker_cleaned'] == 'Franz Josef Jung',
            'speaker_cleaned'] = 'Franz-Josef Jung'
    dfp.loc[dfp['speaker_cleaned'] == 'Aydan Özoğuz',
            'speaker_cleaned'] = 'Aydan Özoguz'
    dfp.loc[dfp['speaker_cleaned'] == 'Sevim Dağdelen',
            'speaker_cleaned'] = 'Sevim Dagdelen'
    dfp.loc[dfp['speaker_cleaned'] == 'Jan Ralf Nolte',
            'speaker_cleaned'] = 'Jan Nolte'
    dfp.loc[dfp['speaker_cleaned'] == 'Alexander Graf Graf Lambsdorff',
            'speaker_cleaned'] = 'Alexander Graf Lambsdorff'
    dfp.loc[dfp['speaker_cleaned'] == 'Michael Georg Link',
            'speaker_cleaned'] = 'Michael Link'
    dfp.loc[dfp['speaker_cleaned'] == 'Eberhardt Alexander Gauland',
            'speaker_cleaned'] = 'Alexander Gauland'
    dfp.loc[dfp['speaker_cleaned'] == 'Fabio De Masi',
            'speaker_cleaned'] = 'Fabio de Masi'
    dfp.loc[dfp['speaker_cleaned'] == 'Ulrich Oehme',
            'speaker_cleaned'] = 'Ulrich Öhme'
    dfp.loc[dfp['speaker_cleaned'] == 'Armin-Paulus Hampel',
            'speaker_cleaned'] = 'Armin Paul Hampel'
    dfp.loc[dfp['speaker_cleaned'] == 'Johann David Wadephul',
            'speaker_cleaned'] = 'Johann Wadephul'
    dfp.loc[dfp['speaker_cleaned'] == 'Joana Eleonora Cotar',
            'speaker_cleaned'] = 'Joana Cotar'
    dfp.loc[dfp['speaker_cleaned'] == 'Sonja Amalie Steffen',
            'speaker_cleaned'] = 'Sonja Steffen'
    dfp.loc[dfp['speaker_cleaned'] == 'Konstantin Elias Kuhle',
            'speaker_cleaned'] = 'Konstantin Kuhle'
    dfp.loc[dfp['speaker_cleaned'] == 'Roman Johannes Reusch',
            'speaker_cleaned'] = 'Roman Reusch'
    dfp.loc[dfp['speaker_cleaned'] == 'Gero Clemens Hocker',
            'speaker_cleaned'] = 'Gero Hocker'
    dfp.loc[dfp['speaker_cleaned'] == 'Ali',
            'speaker_cleaned'] = 'Amira Mohamed Ali'
    dfp.loc[dfp['speaker_cleaned'] == 'Christian Freiherr von Freiherr Stetten',
            'speaker_cleaned'] = 'Christian Freiherr von Stetten'
    dfp.loc[dfp['speaker_cleaned'] == 'Tobias Matthias Peterka',
            'speaker_cleaned'] = 'Tobias Peterka'
    dfp.loc[dfp['speaker_cleaned'] == 'Mariana Iris Harder-Kühnel',
            'speaker_cleaned'] = 'Mariana Harder-Kühnel'
    dfp.loc[dfp['speaker_cleaned'] == 'Johannes Graf Schraps',
            'speaker_cleaned'] = 'Johannes Schraps'
    dfp.loc[dfp['speaker_cleaned'] == 'Siegbert Droese',
            'speaker_cleaned'] = 'Siegbert Dröse'
    dfp.loc[dfp['speaker_cleaned'] == 'Martin Erwin Renner',
            'speaker_cleaned'] = 'Martin E. Renner'
    dfp.loc[dfp['speaker_cleaned'] == 'Bettina Margarethe Wiesmann',
            'speaker_cleaned'] = 'Bettina Wiesmann '
    dfp.loc[dfp['speaker_cleaned'] == 'Jan Ralf Graf Nolte',
            'speaker_cleaned'] = 'Jan Nolte'
    dfp.loc[dfp['speaker_cleaned'] == 'Gerd Graf Müller',
            'speaker_cleaned'] = 'Gerd Müller'
    dfp.loc[dfp['speaker_cleaned'] == 'Helin Evrim Sommer',
            'speaker_cleaned'] = 'Evrim Sommer'
    dfp.loc[dfp['speaker_cleaned'] == 'Udo Theodor Hemmelgarn',
            'speaker_cleaned'] = 'Udo Hemmelgarn'
    dfp.loc[dfp['speaker_cleaned'] == 'Eva-Maria Elisabeth Schreiber',
            'speaker_cleaned'] = 'Eva Schreiber'
    dfp.loc[dfp['speaker_cleaned'] == 'Norbert Maria Altenkamp',
            'speaker_cleaned'] = 'Norbert Altenkamp'
    dfp.loc[dfp['speaker_cleaned'] == 'Katharina Graf Dröge',
            'speaker_cleaned'] = 'Katharina Dröge'
    dfp.loc[dfp['speaker_cleaned'] == 'Britta Katharina Dassler',
            'speaker_cleaned'] = 'Britta Dassler'
    dfp.loc[dfp['speaker_cleaned'] == 'Michael Graf Leutert',
            'speaker_cleaned'] = 'Michael Leutert'
    dfp.loc[dfp['speaker_cleaned'] == 'Eva-Maria Schreiber',
            'speaker_cleaned'] = 'Eva Schreiber'
    dfp.loc[dfp['speaker_cleaned'] == 'Jens Graf Spahn',
            'speaker_cleaned'] = 'Jens Spahn'
    dfp.loc[dfp['speaker_cleaned'] == 'Rolf Graf Mützenich',
            'speaker_cleaned'] = 'Rolf Mützenich'
    dfp.loc[dfp['speaker_cleaned'] == 'Paul Viktor Podolay',
            'speaker_cleaned'] = 'Paul Podolay'
    dfp.loc[dfp['speaker_cleaned'] == 'Martin Graf Hebner',
            'speaker_cleaned'] = 'Martin Hebner'
    dfp.loc[dfp['speaker_cleaned'] == 'Albert H. Weiler',
            'speaker_cleaned'] = 'Albert Weiler'
    dfp.loc[dfp['speaker_cleaned'] == 'Jens Graf Kestner',
            'speaker_cleaned'] = 'Jens Kestner'
    dfp.loc[dfp['speaker_cleaned'] == 'Heidrun Bluhm-Förster',
            'speaker_cleaned'] = 'Heidrun Bluhm'
    dfp.loc[dfp['speaker_cleaned'] == 'Elvan Korkmaz-Emre',
            'speaker_cleaned'] = 'Elvan Korkmaz'
    dfp.loc[dfp['speaker_cleaned'] == 'Katharina Kloke',
            'speaker_cleaned'] = 'katharina willkomm'
    dfp.loc[dfp['speaker_cleaned'] == 'in der beek',
            'speaker_cleaned'] = 'olaf in der beek'

    dfp = dfp.rename(columns={"speaker_cleaned": "name_res"})

    return dfp


# %%


#%%
def create_all_lookups():
    dir = 'res/beta/lookups'
    token2doc = defaultdict(list)
    doc2freq = defaultdict(lambda: defaultdict(int))
    freq = defaultdict(int)
    akteur2freq = defaultdict(lambda: defaultdict(int))
    aggregates = defaultdict(lambda: defaultdict(int))

    q = Document.query.filter(Document.corpus.like('plenar')).all()
    print('Number of Documents: ', len(q))
    # q = sample(q, 1000)
    docs = MyCorpus(q)

    for doc, label in zip(doc_gen(q), tqdm(q)):
        for token in doc:
            token2doc[token].append(label)
            doc2freq[label][token] += 1
            freq[token] += 1
            akteur2freq[label.autor.name][token] += 1
            aggregates[label.autor.party][token] += 1
            aggregates[label.autor.name][token] += 1
            aggregates[label.autor.party]['_sum'] +=1
            aggregates[label.autor.name]['_sum'] += 1

    with open(f'{dir}/t2d.pkl', 'wb') as handle:
        pickle.dump(dict(token2doc), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{dir}/d2f.pkl', 'wb') as handle:
        pickle.dump(dict(doc2freq), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{dir}/freq.pkl', 'wb') as handle:
        pickle.dump(dict(freq), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{dir}/a2f.pkl', 'wb') as handle:
        pickle.dump(dict(akteur2freq), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{dir}/agg.pkl', 'wb') as handle:
        pickle.dump(dict(aggregates), handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# create_all_lookups()

#%%
# dir = 'beta'
# agggregates = pd.read_pickle(f'res/{dir}/lookups/agg.pkl')
# token2doc = pd.read_pickle(f'res/{dir}/lookups/t2d.pkl')
# doc2freq = pd.read_pickle(f'res/{dir}/lookups/d2f.pkl')
# freq = pd.read_pickle(f'res/{dir}/lookups/freq.pkl')
# akteur2freq = pd.read_pickle(f'res/{dir}/lookups/a2f.pkl')

#%%
# create co-occurrence matrix
def __co_occurrence(sentences, total_len, window_size):
    d = defaultdict(int)
    vocab = set()
    freq = defaultdict(int)
    for text in tqdm(sentences, total=total_len):
        # preprocessing (use tokenizer instead)
        # text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            freq[token] += 1
            start = i-window_size
            end = i+1+window_size
            if start < 0:
                start = 0
            if end > len(text) + 1:
                end = len(text) + 1
            next_token = text[start : end]
            next_token.remove(token)
            # print(token, next_token)
            for t in next_token:
                key = tuple(sorted([t, token]))
                d[key] += 1

    for k, v in freq.items():
        if v <= 5:
            vocab.remove(k)
            for key in list(d.keys()):
                if k in key:
                    del d[key]
    del freq


    # dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in tqdm(d.items()):
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df

def create_cooccurrence_per_party(dir):
    for party in db.session.execute(
    "SELECT DISTINCT party FROM akteure"
    ):
        party = party[0]
        print(party)
        q = Document.query.filter(and_(Document.autor.has(Akteur.party == party), Document.corpus.like('plenar'))).all()
        # q = sample(q, 10)
        docs = MyCorpus(q)
        vocab, coo = create_cooccurrence_matrix(dir, docs, 5)
        id = {v: k for k, v in vocab.items()}

        with open(f'res/{dir}/coo/coo_{party}.pkl', 'wb') as handle:
            pickle.dump(coo, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'res/{dir}/coo/vocab_{party}.pkl', 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'res/{dir}/coo/id_{party}.pkl', 'wb') as handle:
            pickle.dump(id, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
# coo.loc['islam'].sort_values(ascending=False)[:20]

#%%
# better alternative
from scipy.sparse import coo_matrix

def create_cooccurrence_matrix(docs, window_size):
    vocabulary={}
    data=[]
    row=[]
    col=[]

    # for tokens in docs:
    #     for pos,token in enumerate(tokens):

    #new from here:
    for doc in docs:
        tokens = pipe(doc)
        for pos, token in enumerate(tokens):
            i=vocabulary.setdefault(token,len(vocabulary))
            start=max(0,pos-window_size)
            end=min(len(tokens),pos+window_size+1)
            for pos2 in range(start,end):
                if pos2==pos:
                    continue
                j=vocabulary.setdefault(tokens[pos2],len(vocabulary))
                data.append(1.); row.append(i); col.append(j);
    cooccurrence_matrix=coo_matrix((data,(row,col)))
    return vocabulary,cooccurrence_matrix


# use pipe to generate tokens from docs
from germalemma import GermaLemma

def pipe(label):
    lemmatizer = GermaLemma()

    # from src.d01_ana.analysis import load_data, gendocs
    def lemma_getter(token):
        try:
            return lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    nlp = spacy.load("de_core_news_lg")
    doc = nlp(label.text)
    res = []

    for i, sent in enumerate(doc.sents):
        for j, token in enumerate(sent):
            Token.set_extension('lemma', getter=lemma_getter, force=True)
            if not token.is_punct and not token.is_digit and not token.is_space:
                tok = token._.lemma.lower()
                tok = tok.replace('.', '')
                res.append(tok)

    return res

# def clean_corpus(text_gen):
#     def remove_URL(sample):
#         """Remove URLs from a sample string"""
#         pattern=r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))"""
#         # return re.sub(r"http\S+", "", sample)
#         return re.sub(pattern, "", sample)
#     # PATH = 'data/corpus'
#     corpus = Corpus()
#     for doc in text_gen:
#     # corpus.add_folder(PATH + '/presse')
#         corpus.add_doc(str(doc.id), remove_URL(doc.text))
#         # corpus.add_doc(str(doc.id), doc.text)

#     # corpus = Corpus.from_folder(PATH + '/plenar', encoding='utf8')
#     # corpus = Corpus.from_folder(PATH + '/plenar')
#     # corpus.add_folder(PATH + '/presse')
#     # corpus.add_folder(PATH + '/twitter')

#     doc_labels = corpus.get_doc_labels(sort=True)

#     table_umlauts = {"ÃŸ": "ß", "ãÿ": "ß", "ã¤": "ä", "ã¼": "ü", "ã¶": "ö", 'Ã„': 'Ä', "Ãœ": "Ü", "Ã–": "Ö", 'â‚¬': '€'}

#     table_chars = {';': '.', '$': '', '?': '.', '!': '.', ':':'.', '@': '', '#': ''}
#     left = corpus.unique_characters - set(string.printable)
#     umlauts = ['ä', 'ü', 'ö', 'Ä', 'Ö', 'Ü', 'ß']
#     for um in umlauts:
#         left.discard(um)
#     for char in left:
#         if char not in table_chars:
#             table_chars[char] = ''
#     keep = ['.', ',']
#     for char in string.punctuation:
#         if char not in table_chars and char not in keep:
#             table_chars[char] = ''

#     # print(table_chars)

#     # phrases = {'teilentweetPrint': '', 'Current Page': '', 'Pressekontakt .   CDUCSU  BundestagsfraktionPressestelleTelefon .   030 22752360Fax .       030 22756660Internet .  http . www . cducsu . deEmail .  pressestellecducsu . de OriginalContent von .  CDUCSU  Bundestagsfraktion, übermittelt durch news aktuell': '', }

#     def repl_phrases(doc):
#         for k, v in phrases.items():
#             doc = doc.replace(k,v)
#         return doc

#     def repl_umlauts(doc):
#         for k, v in table_umlauts.items():
#             doc = doc.replace(k,v)
#         return doc

#     def repl_chars(doc):
#         for k, v in table_chars.items():
#             doc = doc.replace(k, v)
#         return doc

#     def repl_nl(doc):
#         doc = doc.replace(r'\n', "")
#         return doc

#     def repl_last(doc):
#         doc = doc.replace('-', ' ')
#         return doc

#     def repl_dot(doc):
#         doc = doc.replace('.', ' . ')
#         return doc

#     def fix_spaces(doc):
#         doc = ' '.join(doc.split())
#         return doc

#     corpus.apply(lambda x: repl_umlauts(x))
#     corpus.apply(lambda x: repl_chars(x))
#     # corpus.apply(lambda x: repl_nl(x))

#     # corpus.replace_characters(del_chars)

#     # correct contractions
#     pttrn_contraction_ws = re.compile(r'(\w+)(\s+)(-\w+)')
#     corpus.apply(lambda t: pttrn_contraction_ws.sub(lambda m: m.group(1) + m.group(3), t))


#     corpus.apply(lambda x: repl_last(x))
#     corpus.apply(lambda x: repl_dot(x))
#     # corpus.apply(lambda x: repl_phrases(x))

#     def remove_dots(sample):
#         pttrn_dots = re.compile(r'(\. ?)+')
#         return re.sub(pttrn_dots, ".", sample)
#     corpus.apply(lambda x: remove_dots(x))



#     corpus.apply(fix_spaces)

#     # delete special chars in tweets:
#     # left = corpus.unique_characters - set(string.printable)
#     # umlauts = ['ä', 'ü', 'ö', 'Ä', 'Ö', 'Ü', 'ß']
#     # for um in umlauts:
#     #     left.discard(um)
#     # left_dict = {d: None for d in left}

#     # corpus.replace_characters(left_dict)


#     # for i in range(500):
#     #     print(corpus[str(i+1)])

#     print('these non-ASCII characters are left:')
#     pprint(corpus.unique_characters - set(string.printable))

#     for label in doc_labels:
#         yield corpus[str(label)]
