#%%
from flask_sqlalchemy import SQLAlchemy
from . import db

class WordAkteur(db.Model):
    __tablename__ = 'word_akteur_link'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'))
    akteur_id = db.Column(db.Integer, db.ForeignKey('akteure.id'))
    freq = db.Column(db.Integer)

class WordParty(db.Model):
    __tablename__ = 'word_party_link'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'))
    party_id = db.Column(db.Integer, db.ForeignKey('parties.id'))
    freq = db.Column(db.Integer)

class Word(db.Model):
    __tablename__ = 'words'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(64))
    freq = db.Column(db.Integer)
    idf = db.Column(db.Integer, nullable=True)
    akteurfrequency = db.relationship('WordAkteur', backref='word', primaryjoin=id == WordAkteur.word_id)

class Doc(db.Model):
    __tablename__ = 'docs'
    id = db.Column(db.Integer, primary_key=True)
    # filename = db.Column(db.String(64), unique=True)
    text = db.Column(db.Text)
    date = db.Column(db.Date)
    corpus = db.Column(db.String(64))
    autor_id = db.Column(db.Integer, db.ForeignKey('akteure.id'))
    # res = db.relationship('Res', backref='result', uselist=False)
    bin = db.Column(db.PickleType, nullable=True)
    arr = db.Column(db.PickleType, nullable=True)
    res = db.Column(db.PickleType, nullable=True)
    post = db.Column(db.PickleType, nullable=True)

    def __repr__(self):
        return f'doc {self.id}'


class Akteur(db.Model):
    __tablename__ = 'akteure'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    party = db.Column(db.String(64))
    id_party = db.Column(db.Integer)
    agw_18 = db.Column(db.String(64), nullable=True)
    agw_19 = db.Column(db.String(64), nullable=True)
    agw_url = db.Column(db.String(64), nullable=True)
    byear = db.Column(db.String(64), nullable=True)
    gender = db.Column(db.String(64), nullable=True)
    education = db.Column(db.String(64), nullable=True)
    election_list = db.Column(db.String(64), nullable=True)
    profile = db.Column(db.PickleType, nullable=True)
    facebook = db.Column(db.PickleType, nullable=True)
    twitter = db.Column(db.PickleType, nullable=True)
    youtube = db.Column(db.PickleType, nullable=True)
    instagram = db.Column(db.PickleType, nullable=True)
    flickr = db.Column(db.PickleType, nullable=True)

    docs = db.relationship('Doc', backref='autor')
    akteurfrequency = db.relationship('WordAkteur', backref='akteur', primaryjoin=id == WordAkteur.akteur_id)

    def __repr__(self):
        return f'akteur {self.name}'


class Party(db.Model):
    __tablename__ = 'parties'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    party = db.Column(db.String(64))
    id_party = db.Column(db.Integer)
    agw_18 = db.Column(db.String(64), nullable=True)
    agw_19 = db.Column(db.String(64), nullable=True)
    agw_url = db.Column(db.String(64), nullable=True)
    byear = db.Column(db.String(64), nullable=True)
    gender = db.Column(db.String(64), nullable=True)
    education = db.Column(db.String(64), nullable=True)
    election_list = db.Column(db.String(64), nullable=True)
    profile = db.Column(db.PickleType, nullable=True)
    facebook = db.Column(db.PickleType, nullable=True)
    twitter = db.Column(db.PickleType, nullable=True)
    youtube = db.Column(db.PickleType, nullable=True)
    instagram = db.Column(db.PickleType, nullable=True)
    flickr = db.Column(db.PickleType, nullable=True)

    # docs_party = db.relationship('Doc', backref='autor')
    partyfrequency = db.relationship('WordParty', backref='party', primaryjoin=id == WordParty.party_id)

    def __repr__(self):
        return f'Party {self.name}'
