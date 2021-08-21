import os
from germalemma import GermaLemma

def create_dictionary(gen_docs, filename):

    def pipe(label):
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

    if os.path.isfile(filename):
        print('File already exists!')
        return

    # create gensim dict & BoW
    lemmatizer = GermaLemma()

    # from src.d01_ana.analysis import load_data, gendocs
    def lemma_getter(token):
        try:
            return lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    # doc_labels = random.sample(doc_labels, 100)

    nlp = spacy.load("de_core_news_lg")

    docs = (pipe(label) for label in gen_docs)
    # tokens = [(token for token in doc) for doc in docs]
    tokens = ((token for token in doc) for doc in docs)
    dictionary = corpora.Dictionary()

    BoW_corpus = [dictionary.doc2bow(token, allow_update=True) for token in tokens]

    dictionary.save(filename)

    return dictionary
