
# coding: utf-8

# In[4]:

import os
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
import urllib.request
import bs4
import sys


# In[5]:

training_files = ['category/entertainment_20.txt', 'category/golf_20.txt', 'category/politics_20.txt']

snowball_stemmer = SnowballStemmer("english")

# stopwords
stop = set(stopwords.words('english'))
stop = stop | set(['"', '--', "''", '``', ',', '.'])


# In[6]:

word_set = set([]) # word global set


# In[7]:

def crawler(url):
    htmlData = urllib.request.urlopen(url)
    bs = bs4.BeautifulSoup(htmlData, 'lxml')
    text = ''# parsed_text
    # parse header
    header = bs.find('h1', 'pg-headline')
    if header is not None:
        text += header.getText()
    # parse image caption
    caption = bs.find('div', 'js-media__caption media__caption el__storyelement__title')
    if caption is not None:
        text += caption.getText()
    # parse image title
    imgTitle = bs.find('div', 'media__caption el__gallery_image-title')
    if imgTitle is not None:
        text += imgTitle.getText()
    # parse paragraphs
    first = bs.find('cite', class_="el-editorial-source")
    if first is not None:
        first.decompose()
    
    text += bs.find('p', 'zn-body__paragraph').getText()
    bodies = bs.findAll('div', 'zn-body__paragraph')
    for b in bodies:
        text += b.getText()
    return text.lower()


# In[8]:

def preprocessing(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tokens = [i for (i,j) in tagged if i not in stop and j in ['NN','NNP','NNPS','NNS']]
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(snowball_stemmer.stem(token))
    return stemmed_tokens


# In[9]:

def preprocessing_with_file(fname):
    with open(os.path.join(os.getcwd(), fname)) as f:
        return preprocessing(f.read())


# In[10]:

def collect_words_from_url_file(wset, fname):
    with open(os.path.join(os.getcwd(), fname)) as f:
        urls = f.readlines()
        for u in urls:
            wset = wset | set(preprocessing(crawler(u.strip())))
    return wset


# In[11]:

def construct_word_list(wlist, fname):
    with open(os.path.join(os.getcwd(), fname)) as f:
        urls = f.readlines()
        for u in urls:
            wlist = wlist + preprocessing(crawler(u.strip()))
    return wlist


# In[12]:

# construct global word set for 60 articles
for fname in training_files:
    word_set = word_set | collect_words_from_url_file(word_set, fname)


# In[13]:

# create vector index
word_set = list(word_set)

# define topic vectors
ent_vec = [0 for i in range(0, len(word_set))]
golf_vec = [0 for i in range(0, len(word_set))]
pol_vec = [0 for i in range(0, len(word_set))]


# In[14]:

# count word frequency
def counting(tokens, vec, global_list):
    for w in tokens:
        if w in global_list:
            vec[global_list.index(w)] += 1
    return vec


# In[15]:

ent_vec = counting(construct_word_list([], training_files[0]), ent_vec, word_set)
golf_vec = counting(construct_word_list([], training_files[1]), golf_vec, word_set)
pol_vec = counting(construct_word_list([], training_files[2]), pol_vec, word_set)


# # Classification

# In[16]:

import numpy as np
import math


# In[17]:

def cosineSimilarity(v1, v2):
    multi = (v1.dot(v2)).sum()
    x = math.sqrt((v1*v1).sum())
    y = math.sqrt((v2*v2).sum())

    result = multi/(x*y)
    return result


# In[18]:

def classify(doc):
    sim_with_ent = cosineSimilarity(np.array(ent_vec), np.array(doc))
    sim_with_golf = cosineSimilarity(np.array(golf_vec), np.array(doc))
    sim_with_pol = cosineSimilarity(np.array(pol_vec), np.array(doc))
    
    print('sim_with_ent: '), print(sim_with_ent)
    print('sim_with_golf: '), print(sim_with_golf)
    print('sim_with_politics: '), print(sim_with_pol)
    
    if sim_with_ent > sim_with_golf and sim_with_ent > sim_with_pol:
        return 'entertainment'
    elif sim_with_golf > sim_with_ent and sim_with_golf > sim_with_pol:
        return 'golf'
    elif sim_with_pol > sim_with_ent and sim_with_pol > sim_with_ent:
        return 'politics'


# In[19]:

#ent_test_vec = [0 for i in range(0, len(word_set))]
#golf_test_vec = [0 for i in range(0, len(word_set))]
#pol_test_vec = [0 for i in range(0, len(word_set))]


# In[20]:

#ent_test_vec = counting(preprocessing_with_file(test_files[0]), ent_test_vec, word_set)
#golf_test_vec = counting(preprocessing_with_file(test_files[1]), golf_test_vec, word_set)
#pol_test_vec = counting(preprocessing_with_file(test_files[2]), pol_test_vec, word_set)


# In[21]:

#print('entertainment is classified as : ')
#print (classify(ent_test_vec)), print ('\n')

#print('golf is classified as : ')
#print (classify(golf_test_vec)), print ('\n')

#print('politics is classified as : ')
#print (classify(pol_test_vec)), print('\n')


# In[22]:

#ent_test_vec = counting(preprocessing_with_file(test_files[0]), ent_test_vec, word_set)
test_vec = [0 for i in range(0, len(word_set))]
test_vec = counting(preprocessing_with_file(sys.argv[1]), test_vec, word_set)

# print result
print(sys.argv[1], ' is classified as : ', classify(test_vec), '\n')


# # Visualization

# In[ ]:

import matplotlib.pyplot as plt

x = [i for i in range(0, len(word_set))]
freq_list = [(x,y) for x,y in zip(x, ent_vec)]
freq_list.sort(key=lambda x:x[1])
freq_list = freq_list[-5:]

WordOrder = [5,4,3,2,1]
WordFrequency = [x[1] for x in freq_list]

LABELS = [word_set[x[0]] for x in freq_list]

plt.bar(WordOrder, WordFrequency, align='center')
plt.xticks(WordOrder, LABELS)
plt.show()


# In[ ]:

x = [i for i in range(0, len(word_set))]
freq_list = [(x,y) for x,y in zip(x, golf_vec)]
freq_list.sort(key=lambda x:x[1])
freq_list = freq_list[-5:]

WordOrder = [5,4,3,2,1]
WordFrequency = [x[1] for x in freq_list]

LABELS = [word_set[x[0]] for x in freq_list]

plt.bar(WordOrder, WordFrequency, align='center')
plt.xticks(WordOrder, LABELS)
plt.show()


# In[ ]:

x = [i for i in range(0, len(word_set))]
freq_list = [(x,y) for x,y in zip(x, pol_vec)]
freq_list.sort(key=lambda x:x[1])
freq_list = freq_list[-5:]

WordOrder = [5,4,3,2,1]
WordFrequency = [x[1] for x in freq_list]

LABELS = [word_set[x[0]] for x in freq_list]

plt.bar(WordOrder, WordFrequency, align='center')
plt.xticks(WordOrder, LABELS)
plt.show()


# In[ ]:



