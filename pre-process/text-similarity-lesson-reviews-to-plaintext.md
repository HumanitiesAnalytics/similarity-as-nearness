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
import glob
import json 
from bs4 import BeautifulSoup
import pandas as pd 
```

```python
txt_files = glob.glob('../book-reviews/txt/*.txt')
txts = []
for i in txt_files:
    with open(i) as f:
        this_txt = f.read()
        txts.append(this_txt)
txt_files
```

```python
html_files = glob.glob('../book-reviews/html/*.html')
html_files
```

```python
soups = []
for i in html_files:
    with open(i) as f:
        this_txt = f.read()
        this_bs = BeautifulSoup(this_txt)
        soups.append(this_bs)
len(soups)
```

```python
atlantic_vonnegut = '\n\n'.join([i.text for i in soups[0].find_all('div', {'class':'reviewBody'})])
#print(atlantic_vonnegut[0:1000])
```

```python
new_yorker_briefly_noted = json.loads(soups[2].find_all('script', {'type':"application/ld+json"})[0].text)['articleBody']
```

```python
nyt_breakfast = '\n'.join([i.text for i in soups[1].find_all('div', {'class':'StoryBodyCompanionColumn'})])
```

```python
nora_sayre_1 = [i.text for i in soups[3].find_all('div', {'class':'StoryBodyCompanionColumn'})[0].find_all('p')]
```

```python
nora_sayre_2 = [i.text for i in soups[3].find_all('div', {'class':'StoryBodyCompanionColumn'})[1].find_all('p')]
```

```python
nora_sayre = '\n\n'.join(nora_sayre_1 + nora_sayre_2)
#print(nora_sayre[0:1000])
```

```python
# clean up metadata and data for each 
full_text_raw = [atlantic_vonnegut, new_yorker_briefly_noted, nora_sayre]
for e, i in enumerate(html_files):
    new_name = i.lower().replace('html', 'txt')
    print(new_name)
    # lines below are commented out because this is a one-time process 
    # with open(new_name, 'a') as f:
    #    f.write(full_text_raw[e])
```

```python
txt_files = glob.glob('../book-reviews/txt/*.txt')
txts = []
for i in txt_files:
    with open(i) as f:
        this_txt = f.read()
        txts.append(this_txt)

# this ordering could change and might need to be edited if you wanted to add txts to the metadata
txt_files[4], txt_files[1], txt_files[0], txt_files[3], txt_files[5], txt_files[2]
```

```python
# make metadata for txts 
df = pd.DataFrame()
df['periodical_title'] = ['New Yorker', 'The Atlantic', 'The New York Times', 'The New York Times', 'The New York Times', 'Computer Gaming World']
df['review_type'] = ['multi-work', 'single-work', 'single-work', 'single-work', 'single-work', 'single-work']
df['review_author'] = ['Unsigned', 'Richard Todd', 'Nora Sayre', 'Nora Sayre', 'Nora Sayre', 'Charles Ardai']
df['review_pub_date'] = ['04/11/1977', '05/01/1973', '10/05/1969', '08/30/1970', '05/13/1973', '02/01/1994']
df['reviewed_book_title'] = ['multiple', 'Breakfast of Champions', 'The Queen Was in the Garbage', "Enid Bagnold's Autobiography", 'Breakfast of Champions', 'Sam & Max Hit the Road' ] 
df['reviewed_book_author'] = ['multiple', 'Kurt Vonnegut', 'Lila Karp', 'Enid Bagnold', 'Kurt Vonnegut', 'LucasArts']
df['review_file_path'] = txt_files[4], txt_files[1], txt_files[0], txt_files[3], txt_files[5], txt_files[2]
df.to_csv('../review_metadata.csv')
```

```python
df
```

```python

```
