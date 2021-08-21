from app import db
from app.models import Doc, Akteur, Word, WordAkteur, WordParty
import pandas as pd
from sqlalchemy.types import Integer, Text, String, DateTime, PickleType, Boolean, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.config import config
import glob
# from app.src.d00_utils import clean_corpus
import gc
from tqdm import tqdm
from collections import defaultdict
from sqlalchemy import func
from app.src.d00_utils.helper import get_main_dir, get_data_dir
import os

DIR_META = os.path.join(get_data_dir(), "mdbs_metadata.json")
DIR_CORPUS = os.path.join(get_data_dir(), "corpus")

def gendocs(label, corpus):
    with open(f"{DIR_CORPUS}/{corpus}/{label}", "r") as text_file:
        return text_file.read()


def create_session(config_name):
    uri = config[config_name].SQLALCHEMY_DATABASE_URI
    engine = create_engine(uri, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def pop_docs():
    corpora = ['plenar', 'presse', 'twitter']
    # corpora = ['plenar']
    for corpus in corpora:
        df = pd.read_json(f'{DIR_CORPUS}/{corpus}_meta.json', orient='index')
        mapping = {}
        # df["date"] = pd.to_datetime(df["datum"], unit="ms", errors="ignore")

        # DROP cases without Date! (27 in plenar, 185 in twitter)
        df = df.dropna(subset=['datum'])

        # convert to datetime
        # df["date"] = pd.to_datetime(df["datum"])

        # session = create_session(config_name)
        # for name in df.name_res.unique:
            # mapping[name] = None

        for instance in db.session.query(Akteur):
            mapping[instance.name] = instance.id

        for label, date, name in zip(df.index, df.date, df.name_res):
            print(label, date, name)
            text = gendocs(label, corpus)
            akteur_id = mapping[name.lower()]
            doc = Doc(text=text, date=date, autor=Akteur.query.filter_by(id=akteur_id).first(), corpus=corpus)
            db.session.add(doc)


def pop_akteure():
    df = pd.read_json(DIR_META, orient='index')
    # df = pd.read_pickle(DIR_META)
    # df = df[~df.name.str.contains("partei")]
    # db.create_all()
    # db.session.commit()

    # df.rename(columns={'name_res': 'name', 'profile_url': 'agw_url', 'birth_year': 'byear'}, inplace=True)
    # df.drop(columns=['first_name', 'last_name', 'id_mdb', 'identifier', 'profiles_count'], inplace=True)
    # df['byear'] = pd.to_numeric(df['byear'])

    table_name = 'akteure'

    df.to_sql(
        table_name,
        db.engine,
        if_exists='append',
        index=False,
        chunksize=500,
        dtype={
            'id': Integer,
            'name': String,
            'party': String,
            'id_party': Integer,
            'agw_18': String,
            'agw_19': String,
            'agw_18': String,
            'byear': String,
            'gender': String,
            'education': String,
            'election_list': String,
            'profile': PickleType,
            'facebook': PickleType,
            'twitter': PickleType,
            'youtube': PickleType,
            'instagram': PickleType,
            'flickr': PickleType
        }
    )
    # db.session.commit()

def pop_parties():
    # df = pd.read_pickle(DIR_META)
    df = pd.read_json(DIR_META, orient='index')
    df = df[df.name.str.contains("partei")]
    df.name = df.party
    df = df.append({'id_party': 0, 'name':'Parteilos', 'party': 'Parteilos'}, ignore_index=True)
    # db.create_all()
    # db.session.commit()

    # df.rename(columns={'name_res': 'name', 'profile_url': 'agw_url', 'birth_year': 'byear'}, inplace=True)
    # df.drop(columns=['first_name', 'last_name', 'id_mdb', 'identifier', 'profiles_count'], inplace=True)
    # df['byear'] = pd.to_numeric(df['byear'])

    table_name = 'parties'

    df.to_sql(
        table_name,
        db.engine,
        if_exists='append',
        index=False,
        chunksize=500,
        dtype={
            'id': Integer,
            'name': String,
            'party': String,
            'id_party': Integer,
            'agw_18': String,
            'agw_19': String,
            'agw_18': String,
            'byear': String,
            'gender': String,
            'education': String,
            'election_list': String,
            'profile': PickleType,
            'facebook': PickleType,
            'twitter': PickleType,
            'youtube': PickleType,
            'instagram': PickleType,
            'flickr': PickleType
        }
    )

def pop_words():
    dir = 'beta'
    aggs = pd.read_pickle(f'res/{dir}/lookups/agg.pkl')
    # token2doc = pd.read_pickle(f'res/{dir}/lookups/t2d.pkl')
    # doc2freq = pd.read_pickle(f'res/{dir}/lookups/d2f.pkl')
    freq = pd.read_pickle(f'res/{dir}/lookups/freq.pkl')
    akteur2freq = pd.read_pickle(f'res/{dir}/lookups/a2f.pkl')

    # populate words
    res = []
    for word in freq:
        w = Word(text=word, freq=freq[word])
        res.append(w)
    db.session.add_all(res)


def pop_rel_akteure():
    # populate relationship: word = akteur
    akts = Akteur.query.all()
    words = Word.query.all()


    # populate Akteure
    for i, akt in enumerate(akts[:]):
        # print(i, akt.name)
        if akt.name in akteur2freq.keys():
            res = []
            for w in words:
                frequency = akteur2freq[akt.name][w.text]
                if frequency != 0:
                    link = WordAkteur(word_id=w.id, akteur_id=akt.id, freq=frequency)
                    res.append(link)

            db.session.add_all(res)
        if i % 50 == 0 and i > 0:
            db.session.commit()
            print('last commit at index: ', akts.index(akt))
            gc.collect()



def pop_rel_parties():
    # populate partiy-frequencies

    partie_names = [i[0] for i in db.session.execute("SELECT DISTINCT party FROM akteure;")]
    parties = Party.query.filter(Party.name.in_(partie_names)).all()

    freqs_per_party = db.session.query(func.sum(WordAkteur.freq), \
        Akteur.party, \
        WordAkteur.word_id).\
        join(WordAkteur).\
        group_by(Akteur.party, WordAkteur.word_id).all()

    res = []
    for tup in tqdm(freqs_per_party):
        for party in parties:
            if tup[1] == party.name:
                link = WordParty(word_id=tup[2], party_id=party.id, freq=tup[0])
                res.append(link)

    db.session.add_all(res)


def init_database():
    from ana import db
    from ana.models import Doc, Akteur
    # from database_v2 import pop_akteure, pop_docs
    db.create_all()
    pop_akteure()
    pop_parties()
    db.session.commit()
    # pop_docs()
    # db.session.commit()


if __name__ == '__main__':
    from ana import db
    from ana.models import Doc, Akteur
    # from database import pop_akteure, pop_docs, pop_tweets, clean_tweets
    db.create_all()
    pop_akteure()
    db.session.commit()
    pop_docs()
    db.session.commit()
    pop_tweets()
    clean_tweets()
    db.session.commit()
    db.session.close()
