---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd 
from sklearn.feature_extraction import DictVectorizer
import spacy
from collections import Counter
import pronouncing
import numpy as np 
```

```python
nlp = spacy.load("en_core_web_sm")

def make_label(row):
    pre = 'CW_'
    if row['pos'] in ['ADP', 'DET', 'PRON', 'AUX', 'CCONJ', 'SCONJ']:
        pre = 'FW_'
    if row['pos'] == 'ADJ':
        pre = 'CW_ADJ_'
    token = row['token_lower']
    return pre + token

# count syllables 
def try_count_syll(row):
    try:
        pronunciations = pronouncing.phones_for_word(row['token_lower'])
        syllables = [ pronouncing.syllable_count(i) for i in pronunciations ] 
        return np.median(syllables)
    except:
        return None
```

```python
df = pd.read_csv('../review_metadata.csv', index_col=0)

txts = []
for e, row in df.iterrows():
    with open(row['review_file_path']) as f:
        txt = f.read().replace('—', ' ').replace('-', ' ').replace('‐', ' ')
        txts.append(txt)
len(txts)
```

```python
docs = [nlp(i) for i in txts]
type(docs[0])
```

```python
data = []
for e, doc in enumerate(docs):
    token_pos = [(token.text, token.pos_, e) for token in doc]
    data.extend(token_pos)
df_tokens = pd.DataFrame(data, columns=['token', 'pos', 'doc_id'])
df_tokens['token_lower'] = df_tokens['token'].str.lower()
df_tokens.head(10)
```

```python
# remove punctuation 
token_pos_grouped = df_tokens.loc[~df_tokens['pos'].isin(['SPACE', 'PUNCT', 'PART', 'NUM', 'SYM'])].groupby(['doc_id', 'token_lower', 'pos']).count().reset_index().sort_values(by=['doc_id', 'token'], ascending=[True, False])

# remove stray symbols (&, ♦, �)
token_pos_culled = token_pos_grouped.loc[~token_pos_grouped['token_lower'].isin(['&', '♦', '�'])]
token_pos_culled.head(10)
```

```python
# make column names 
token_pos_culled['label'] = token_pos_culled.apply(make_label, axis= 1)

#token_pos_culled.loc[token_pos_culled['label'].str.startswith('CW_ADJ')]

token_pos_culled = token_pos_culled.loc[token_pos_culled['token_lower'].str.isalpha()]

# check for repeats
token_pos_culled.groupby('label').sum()
```

```python
# vectorize 
lod = []
for i in range(6):
    this_dict = token_pos_culled.loc[token_pos_culled['doc_id'] == i][['label', 'token']].set_index('label').to_dict()['token']
    lod.append(this_dict)
Counter(lod[0]).most_common(12)
```

```python
v = DictVectorizer(sparse=False)
X = v.fit_transform(lod)
# five reviews, 2442 unique words (tokens)
matrix = pd.DataFrame(X, columns=v.feature_names_)
matrix
```

```python
token_pos_culled['syllables'] = token_pos_culled.apply(try_count_syll, axis=1)

# add columns for sentence length and syllables 
matrix['syllables'] = token_pos_culled.groupby('doc_id').mean(numeric_only=True)['syllables']

sentences_all = []
for doc in docs:
    sentences = []
    for i in doc.sents:
        sentences.append(len(i))
    sentences_all.append(np.mean(sentences))
matrix['sentence_len'] = sentences_all
```

```python
df_hapaxes = pd.DataFrame(matrix.astype(bool).sum(axis=0), columns=['count'])
hapaxes = df_hapaxes.loc[df_hapaxes['count'] <= 1].index.to_list()
len(hapaxes)
```

```python
matrix_no_hapax = matrix[[i for i in matrix.columns if i not in hapaxes]]
```

```python
# cull or norm these upstream and one doc words
for i in v.feature_names_:
    if not i.replace('_', '').isalpha():
        print(i)
```

```python
# save to csv (for R) 
matrix.to_csv('../csv/review_features.csv')
matrix_no_hapax.to_csv('../csv/review_features_no_hapax.csv')
```
