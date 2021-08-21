import pytest
import os
import pickle
from app import create_app, db, config
from app.models import Doc, Akteur
from pandas import to_datetime
from copy import deepcopy

def test_config(test_app_base):
    res = test_app_base.application.config.get('LOGFILE')
    assert res == 'logfile_test.log', 'config not found'
    assert test_app_base.application.config.get('SQLALCHEMY_DATABASE_URI') == 'sqlite://'

def test_db(test_app):
    test_doc = Doc.query.filter_by(id=1).first()
    tdoc_ger_pos = Doc.query.filter_by(id=2).first()
    tdoc_afd = Doc.query.filter_by(id=3).first()
    tdoc_tweet = Doc.query.filter_by(id=4).first()
    assert test_app.application.config.get('SQLALCHEMY_DATABASE_URI') == 'sqlite://'
    assert db.engine.url.__to_string__() == 'sqlite://'
    assert test_doc.autor.name == 'donald trump'
    assert test_doc.autor.id_party == 42
    assert Akteur.query.all() != []
    assert tdoc_afd.text.startswith('Sehr verehrte')
    assert tdoc_tweet.corpus == 'twitter'

def test_expecting_wrong_akteur(test_app):
    test_doc = Doc.query.filter_by(id=1).first()
    with pytest.raises(AssertionError) as excinfo:
        assert test_doc.autor.name == 'angela merkel'

