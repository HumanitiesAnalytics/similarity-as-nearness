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
df
```

```python
root = """Meanwhile, there are a few comforts in the Vonnegut universe: sex, occasional travel to other planets, booze (even though it's really “yeast excrement”), the love of a good dog. (While it's acknowledged that everyone needs “uncritical love,” that essence seems to be getting scarcer of late.) But due to the ease with which Vonnegut can pitch his people into situations that are “complex, tragic and laughable,” he is still our funniest pessimist, a magician of misery and farce. 
Naturally, “Breakfast of Champions” is laced with lunacies: a motel bedroom smells of raspberries from the disinfectant and roach killer which the management uses; a Midwestern Festival of the Arts is “postponed because of madness"; the author himself is almost savaged by a murderous Doberman pinscher he had tried (but failed) to cut from an earlier version of his own book. As usual, the deadpan derision is shot with sympathy."""

candidate_1 = """In the last sentence you begin to hear one of the purposes of this style: to see life in all its mundane brutality. This is autism with a conscience, often a quite specific social conscience. And truly it is hard to be for what Vonnegut is against, including, as it does, slavery, jingoism, racism, commercial greed, ecological disaster. He thinks that the names of American cars and corporations are foolish to the point of obscenity. Holiday Inns are droll places indeed. The American Indian has been given a bad deal. I’d agree. Vonnegut brings a remarkable air of discovery to these themes, the pretense that no one has quite seen before the stark outlines of our hypocrisy. It is a part of his appeal for his readers that I never understood: the banality, the nearly Kiwanian subtlety of his social criticisms they are boosterism in reverse. Here is Vonnegut on the distance between reality and American ideals: “His high school was named after a slave owner who was one of the world’s greatest theoreticians on the subject of human liberty.”
If you say that this wit is easy sophomoric cynicism, though, you have to allow that now Vonnegut says so too."""

candidate_2 = """Perhaps novels about the two sexes are becoming as rare as a workable transit system. Currently, each gender seems to be peering at the other through a small knothole ... reporting experience strictly from one side of the fence, or the rails, or the sheets or whatever barrier is most convenient. ... So many novels suggest that even the battle of the sexes has yielded to their mutual isolation. But naturally most writers can record only what they know    which is frustration. The rondo theme is that both sexes have been condemned, jailed and left without food by the other. It hardly sounds like a revolution."""

candidate_3 = """There is a reason that mainstream is more popular than underground, no matter what field you are discussing: it is more fun and less “in your face”; it’s calculated to entertain rather than to disconcert. ... Self-referential jokes, sneering remarks, deadpan derision, sarcasm, and ridicule can all be funny, but only to a point. Push the point too far and you can give the impression of the party guest trying too hard to be “on,” the comedian who doesn’t know when to let up, or the teenage boor with a wisecrack for every situation. ... It is probably not very useful to psychoanalyze cartoon characters, but Sam and Max are a pair of aggressive little buggers who vent gobs of latent hostility and alienation in their obnoxious running commentary on everything they encounter. They are constantly egging each other on, trying to top each other in cleverness and sang froid, and it tries one's patience."""

excerpts = [nlp(i) for i in [root, candidate_1, candidate_2, candidate_3 ]]
excerpts[0]
```

```python
def preprocess(docs):
    data = []
    for e, doc in enumerate(docs):
        token_pos = [(token.text, token.pos_, e) for token in doc]
        data.extend(token_pos)
    df_tokens = pd.DataFrame(data, columns=['token', 'pos', 'doc_id'])
    df_tokens['token_lower'] = df_tokens['token'].str.lower()
    
    # remove punctuation 
    token_pos_grouped = df_tokens.loc[~df_tokens['pos'].isin(['SPACE', 'PUNCT', 'PART', 'NUM', 'SYM'])].groupby(['doc_id', 'token_lower', 'pos']).count().reset_index().sort_values(by=['doc_id', 'token'], ascending=[True, False])

    # remove stray symbols (&, ♦, �)
    token_pos_culled = token_pos_grouped.loc[~token_pos_grouped['token_lower'].isin(['&', '♦', '�'])]

    # make column names 
    token_pos_culled['label'] = token_pos_culled.apply(make_label, axis= 1)

    token_pos_culled = token_pos_culled.loc[token_pos_culled['token_lower'].str.isalpha()]
    
    # check for repeats
    # token_pos_culled.groupby('label').sum()

    # vectorize 
    lod = []
    for i in range(token_pos_culled['doc_id'].max()+1):
        this_dict = token_pos_culled.loc[token_pos_culled['doc_id'] == i][['label', 'token']].set_index('label').to_dict()['token']
        lod.append(this_dict)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(lod)
    
    matrix = pd.DataFrame(X, columns=v.feature_names_)

    token_pos_culled['syllables'] = token_pos_culled.apply(try_count_syll, axis=1)

    # add column for syllables 
    matrix['syllables'] = token_pos_culled.groupby('doc_id').mean(numeric_only=True)['syllables']

    sentences_all = []
    for doc in docs:
        sentences = []
        for i in doc.sents:
            sentences.append(len(i))
        sentences_all.append(np.mean(sentences))

    # add column for sentence length
    matrix['sentence_len'] = sentences_all

    df_hapaxes = pd.DataFrame(matrix.astype(bool).sum(axis=0), columns=['count'])
    hapaxes = df_hapaxes.loc[df_hapaxes['count'] <= 1].index.to_list()

    matrix_no_hapax = matrix[[i for i in matrix.columns if i not in hapaxes]]

    return matrix, matrix_no_hapax

matrix, matrix_no_hapax = preprocess(docs)

# save to csv (for R) 
# matrix.to_csv(f'../csv/review_features.csv')
# matrix_no_hapax.to_csv(f'../csv/review_features_no_hapax.csv')
```

```python
excerpt_matrix, excerpt_matrix_no_hapax = preprocess(excerpts)
excerpt_matrix.to_csv(f'../book-reviews/csv/excerpt_review_features.csv')
excerpt_matrix_no_hapax.to_csv(f'../book-reviews/csv/excerpt_review_features_no_hapax.csv')

```

```python
# cull or norm these upstream and one doc words
# for i in v.feature_names_:
# if not i.replace('_', '').isalpha():
    # print(i)
```

```python

```
