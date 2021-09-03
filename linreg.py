# %%
import seaborn as sns
import pandas as pd
from collections import Counter
import re
import random
from gensim.models import tfidfmodel
import math
import pickle
from tqdm import tqdm
import numpy as np
from math import log

# %%
with open('doc_labels_presse.pkl', 'rb') as f:
    doc_labels_presse = pickle.load(f)
with open('doc_labels_twitter.pkl', 'rb') as f:
    doc_labels_twitter = pickle.load(f)
with open('doc_labels_plenar.pkl', 'rb') as f:
    doc_labels_plenar = pickle.load(f)

doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]
dfog = pd.read_csv('/media/philippy/SSD/res_all_0808.csv')

# dfog['len'] = lendocs_fix


# %%
dfval_1 = pd.read_json('/media/philippy/SSD/data/ma/corpus/plenar_meta.json', orient='index')
dfval_2 = pd.read_json('/media/philippy/SSD/data/ma/corpus/presse_meta.json', orient='index')
dfval_3 = pd.read_json('/media/philippy/SSD/data/ma/corpus/twitter_meta.json', orient='index')

dfval = dfval_1.append([dfval_2, dfval_3])

dfval['doc'] = dfval.index

dfval['doc'] = dfval.doc.apply(lambda x: x.split('.')[0])

# fix timestamps
df = dfval.copy()

df['date'] = df.datum
df['date'] = pd.to_datetime(df['date'], unit='ms', errors='ignore')
df['presse_datum_clean'] = pd.to_datetime(df['presse_datum_clean'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'].fillna(df.presse_datum_clean, inplace=True)
df['date'].fillna(df.timestamp, inplace=True)

dfval = df

dfs = dfog.merge(dfval.loc[:, ['date', 'party', 'doc']], how='left', on='doc')

# %%
dfs = dfs.set_index('date').loc['2013-10-01':'2020-01-01']
dfs['date'] = dfs.index

# dfs.reset_index()
# %%
# dft = dft.groupby(by='party').resample('Y').mean().reset_index()
# dft['jDate'] = dft['date'].dt.strftime('%y')
# dft['jDate'] = dft.jDate.astype('int')

# %%
from ast import literal_eval
# c1 = eval(dfs.loc[0, 'volk_counter'])

dfs['volk_counter'] = dfs.apply(lambda row: eval(str(row.volk_counter)), axis=1)
dfs['elite_counter'] = dfs.apply(lambda row: eval(str(row.elite_counter)), axis=1)
dfs['lemma_pop'] = dfs.apply(lambda row: eval(str(row.lemma_pop)), axis=1)

# dfs['lemma_pop'] = dfs.apply(lambda row: Counter([i.strip("'") for i in row.lemma_pop.split(',')]), axis=1)

# %%

dictionary = pickle.load(open('gnsm_dict_all.pkl', 'rb'))

def compute_score(counts, doclen):
    scores = []
    for term in counts:
        df = dictionary.dfs[dictionary.token2id[term.lower()]]
        score = tfidfmodel.df2idf(df, len(doc_labels), log_base=2.0, add=1.0) * counts[term]
        scores.append(score)
    res = sum(scores)/log(doclen+3)
    # if doclen != 0:
    #     res = sum(scores)/(log(doclen))
    #     # res = sum(scores)
    # else:
    #     res = np.nan
    return res
    # if res != 0.0:
    #     return math.log(res/doclen)
    # else:
    #     return math.log(0.0000001/doclen)
    # return math.log(sum(scores)/doclen)


dfs['all_counter'] = dfs.apply(lambda row: {**row.volk_counter, **row.elite_counter}, axis=1)
dfs['score_pop'] = dfs.apply(lambda row: compute_score(row.lemma_pop, row['len']), axis=1)
dfs['score'] = dfs.apply(lambda row: compute_score(row.all_counter, row['len']), axis=1)
dfs['score_volk'] = dfs.apply(lambda row: compute_score(row.volk_counter, row['len']), axis=1)
dfs['score_elite'] = dfs.apply(lambda row: compute_score(row.elite_counter, row['len']), axis=1)

dfs['typ'] = dfs['doc'].apply(lambda row: row.split('_')[0])


def isopp(row):
    if row.party in ['CDU', 'SPD', 'CSU']:
        return 'not_opp'
    else:
        return 'opp'
dfs['opp'] = dfs.apply(lambda row: isopp(row), axis=1)

dfb = dfs.copy()
# %%
dfs = dfb.copy()

# dfs = dfs[dfs['score_pop'] != 0.0]
dfs = dfs[dfs['len'] > 630]
dfs = dfs.dropna(subset=['typ', 'party', 'opp', 'score_pop'])

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(dfs[['score_pop']])
dfs['zscore'] = scaler.transform(dfs[['score_pop']])

# %%
import matplotlib.pyplot as plt

# dfs = dfs[dfs['zscore'] < 2.5]

sns.set_style('whitegrid')

sns.violinplot(x=dfs['party'], y=dfs['zscore'])

# %%
sns.boxplot(x=dfs['party'], y=dfs['zscore'])

# %%
import statsmodels.api as sm
reg = dfs[['typ', 'opp', 'date', 'score_pop', 'party']].dropna()

reg['date'] = reg['date'].dt.strftime('%y')
reg['date'] = reg.date.astype('int')
reg['date'] = reg['date'] - 13

sm.add_constant(reg)

# res = sm.Poisson.from_formula("score_pop ~ C(opp, Treatment('not_opp')) + date + C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit_regularized()
# res = sm.Poisson.from_formula("score_pop ~ date", reg).fit()


# res = sm.Poisson.from_formula("score_pop ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

res = sm.Poisson.from_formula("score_pop ~ C(typ, Treatment('plenar')) + C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.Poisson.from_formula("score_pop ~ C(typ, Treatment('plenar')) + C(party, Treatment('CDU'))", reg, missing='drop').fit()
res.summary()

# %%
# poisson reg typ
import statsmodels.api as sm
reg = dfs[['typ', 'score_pop', 'party']].dropna()

# reg['date'] = reg['date'].dt.strftime('%y')
# reg['date'] = reg.date.astype('int')
# reg['date'] = reg['date'] - 13

sm.add_constant(reg)

# res = sm.Poisson.from_formula("score_pop ~ C(opp, Treatment('not_opp')) + date + C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit_regularized()
# res = sm.Poisson.from_formula("score_pop ~ date", reg).fit()


res = sm.Poisson.from_formula("score_pop ~ C(typ, Treatment('plenar')) + C(party, Treatment('CDU'))", reg, missing='drop').fit()
# res = sm.Poisson.from_formula("score_pop ~ C(party, Treatment('CDU')) + C(typ, Treatment('plenar'))", reg, missing='drop').fit()
res.summary()

# %%
df_monthly = dfs['score_pop'].resample('M').mean()

# %%
start, end = '2013-10', '2019-12'

fig, ax = plt.subplots()
ax.plot(df_monthly.loc[start:end],
marker = '.', linestyle='-', linewidth=0.5, label='per month')
ax.plot(df_year.loc[start:end],
marker='o', markersize=7, linestyle='-', color='orange', label='per year')
ax.set_ylabel('avg. populism')
ax.legend();

# %%
with open('summary_typ.csv', 'w') as fh:
    fh.write(res.summary().as_csv())

# %%
