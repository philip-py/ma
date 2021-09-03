#%%
from pydantic import BaseModel
from typing import List, Optional, Union, Set
from app import create_app, db
from app.models import Akteur
from app.models import Doc as Document
from app.src.d00_utils.preproc import clean_corpus_db

from pathlib import Path
import seaborn as sns
from logzero import setup_logger
import os
import random

os.environ["FLASK_CONFIG"] = 'development'

app = create_app('default')
app.app_context().push()

#%%
docs = db.session.query(Document).filter(Document.corpus == 'presse').all()
docs = random.sample(docs, 100)

# %%
clean_corpus_db(docs)

#%%
# double check docs
docs = (i for i in random.sample(Document.query.all(), 100))
docs = random.sample(Document.query.all(), 100)

#%%
for i in range(10):
	doc = docs[i]
	print(doc.autor, doc.date, doc.autor.party, doc.corpus)
	print(doc.text)
	print()

#%%
