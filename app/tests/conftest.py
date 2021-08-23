import pytest
import os
from app import create_app, db
from app.models import Doc, Akteur
from pandas import to_datetime
from app import config


# test akteur
@pytest.fixture(scope='session')
def testakteur():
    return(Akteur(name='donald trump', party='afd', id_party=42, agw_18='url1', agw_19='url2', agw_url='url3', byear=1950, gender='q', education='celeb', election_list='USA', facebook='www.fb.de', twitter=['https://twitter.com/realDonaldTrump'], youtube=None, instagram=None, flickr=None))

# basic test case
@pytest.fixture(scope='session')
def testdoc_base(testakteur):
    return(Doc(text='Wir sind das Volk . Merkel ist korrupt . Politiker sind nicht schlecht . Die Regierung betrügt die deutschen Menschen am 25. Mai . http://www.twitter.com #%*(@', corpus='plenar', date=to_datetime('2018-01-01'), autor=testakteur))

# afd test case
@pytest.fixture(scope='session')
def testdoc_afd(testakteur):
    BASEDIR = config[os.environ.get('FLASK_CONFIG')].DIR_MAIN
    with open(f'{BASEDIR}/tests/fixtures/afd.txt', "r", encoding='utf-8') as f:
        text = f.read()
    return(Doc(text=text, corpus='plenar', date=to_datetime('2018-01-01'), autor=testakteur))

# specific test-cases:
@pytest.fixture(scope='session')
def testdoc_ger_pos(testakteur):
    return(Doc(text='In Deutschland ist es sehr schön .', corpus='plenar', date=to_datetime('2018-01-01'), autor=testakteur))

@pytest.fixture(scope='session')
def testdoc_tweet(testakteur):
    return(Doc(text='Das ist ein Tweet', corpus='twitter', date=to_datetime('2018-01-01'), autor=testakteur))

@pytest.fixture(scope='session')
def app():
    if os.environ.get('FLASK_CONFIG') != 'testing':
        raise Exception("Wrong FLASK_CONFIG, should be 'testing'")
    app = create_app(os.getenv('FLASK_CONFIG') or 'testing')
    app.app_context().push()
    app.debug = True
    # return app.test_client()
    return app

@pytest.fixture(scope='module')
def test_app_base(app, testakteur, testdoc_base):
    db.create_all()
    test_cases = [testakteur, testdoc_base]
    db.session.add_all(test_cases)
    yield app.test_client()
    db.session.rollback()
    db.session.close()


@pytest.fixture(scope='module')
def test_app(app, testakteur, testdoc_base, testdoc_ger_pos, testdoc_afd, testdoc_tweet):
    db.create_all()
    # db.session.commit()

    test_cases = [testakteur, testdoc_base]

    test_cases.extend([
	    testdoc_ger_pos, testdoc_afd, testdoc_tweet
    ])

    # commit to db
    db.session.add_all(test_cases)
    yield app.test_client()
    db.session.rollback()
    db.session.close()

@pytest.fixture(scope='module')
def test_app_real_case(app, testakteur, testdoc_afd):
    db.create_all()

    test_cases = [testakteur, testdoc_afd]

    db.session.add_all(test_cases)
    yield app.test_client()
    db.session.rollback()
    db.session.close()

