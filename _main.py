#%%
from pydantic import BaseModel
from typing import List, Optional, Union, Set
from app import create_app, db
from app.models import Akteur, Doc as Document
from pathlib import Path
import seaborn as sns
from logzero import setup_logger
import os

os.environ["FLASK_CONFIG"] = 'development'

app = create_app('default')
app.app_context().push()

#%%
# testing the tests
from app.src.d01_ana.analysis import Config, Analysis
from app.config import config
from pandas import to_datetime

# ADD TESTDOCS FIRST, DON'T FORGET TO ROLLBACK!

# def testakteur():
    # return(Akteur(name='donald merkel', party='afd', id_party=42, agw_18='url1', agw_19='url2', agw_url='url3', byear=1950, gender='q', education='celeb', election_list='USA', facebook='www.fb.de', twitter=['https://twitter.com/realDonaldTrump'], youtube=None, instagram=None, flickr=None))

def testdoc_afd():
    BASEDIR = config[os.environ.get('FLASK_CONFIG')].DIR_MAIN
    # testakteur = Akteur(name='donald trump', party='afd', id_party=42, agw_18='url1', agw_19='url2', agw_url='url3', byear=1950, gender='q', education='celeb', election_list='USA', facebook='www.fb.de', twitter=['https://twitter.com/realDonaldTrump'], youtube=None, instagram=None, flickr=None)
    testakteur = db.session.query(Akteur).filter_by(name='donald trump').first()
    with open(f'{BASEDIR}/tests/fixtures/afd.txt', "r", encoding='utf-8') as f:
        text = f.read()
    return(Document(text=text, corpus='plenar', date=to_datetime('2018-01-01'), autor=testakteur))


test_cases = [testdoc_afd()]
db.session.add_all(test_cases)

#%%
docs = db.session.query(Akteur).filter_by(name='donald trump').first()

#%%
settings_analysis = {
'debug': False,
'sample': [docs.docs[-1]],
'clf_model': 'joeddav/xlm-roberta-large-xnli',
'corpus': ['plenar'],
'pipeline': ['extensions', 'sentiment', 'entity', 'res'],
# 'pipeline': ['extensions', 'sentiment', 'entity', 'res']
#     'pipeline': ['extensions', 'sentiment', 'entity', 'res', 'spans', 'clf']
}
for i in range(100):
    content_analysis = Analysis('test', Config(**settings_analysis))
    content_analysis(to_disk=False, to_db=False)

    # res = content_analysis.get_results()
    # print(res.viz[-1])

    res = content_analysis.get_results()
    # print(res)

    for doc in res.viz:
        for hit in doc:
            hit.E = True
            hit.V = True
    res.prepare()
    # res.coding_pop()
    # res.create_df()
    # res.compute_score_spans()
    print(res.top_spans()[0])
    print(res.viz[0][0])

#%%
db.session.rollback()
db.session.close()

#%%

#%%
docs.docs[0].res

#%%


#%%
# count pop_spans
len(res.spans_dict[29761])

#%%
def top_spans(spans_dict, topn=10):
    all_spans = []
    for doc in spans_dict.items():
        for span in doc[1]:
            all_spans.append((doc[0], span, spans_dict[doc[0]][span]))
    all_spans.sort(key=lambda tup: tup[2], reverse=True)
    return all_spans[:topn]

top_spans(res.spans_dict)


#%%
from app import config
from pandas import to_datetime

def load_test_case(BASEDIR):
    with open(f'{BASEDIR}/tests/fixtures/afd.txt', "r") as f:
        afd_case = f.read()
    return afd_case

BASEDIR = config[os.environ.get('FLASK_CONFIG')].DIR_MAIN

# create test_case
test_akteur = Akteur(name='donald trump', party='afd', id_party=42, agw_18='url1', agw_19='url2', agw_url='url3', byear=1950, gender='q', education='celeb', election_list='USA', facebook='www.fb.de', twitter=['https://twitter.com/realDonaldTrump'], youtube=None, instagram=None, flickr=None)
# test_doc = Document(text='Wir sind das Volk . Merkel ist korrupt . Politiker sind nicht schlecht . Die Regierung betrügt die deutschen Menschen am 25. Mai . http://www.twitter.com #%*(@', date=to_datetime('2018-01-01'), autor=test_akteur)
# test_cases = [test_akteur, test_doc]
test_cases = [test_akteur]

# add specific test-cases:
# tdoc_ger_pos = Document(text='In Deutschland ist es sehr schön .', date=to_datetime('2018-01-01'), autor=test_akteur)
tdoc_afd = Document(text=load_test_case(BASEDIR), date=to_datetime('2018-01-01'), autor=test_akteur)
test_cases.extend([
    tdoc_afd
])

# commit to db
db.session.add_all(test_cases)

#%%
db.session.commit()

#%%
db.session.rollback()


# %%
doc = db.session.query(Document).filter_by(id=25575).first()


#%%
from app.config import config

def load_and_clean_test_case():
    # act
    with open('tests/fixtures/afd.txt', "r") as f:
        afd_case = f.read()
    print(afd_case)

load_and_clean_test_case()

#%%
from app import config
BASEDIR = config[os.environ.get('FLASK_CONFIG')].DIR_MAIN
with open(f'{BASEDIR}/tests/fixtures/afd.txt', "r", encoding="utf-8") as f:
    text = f.read()
print(text)
