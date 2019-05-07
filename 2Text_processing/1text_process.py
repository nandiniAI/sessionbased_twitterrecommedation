import pandas as pd
import re
import string
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk

pd.set_option('display.max_colwidth', -1)  
pd.set_option('display.max_rows', None)
colnames = ['created_at','screen_name', 'text']
data = pd.read_csv('database.csv', sep=',', names=colnames)
text = data.text
text = re.sub(r'[0-9]+', '',str(text)) #remove numbers
text=re.sub(r'\\x([a-z])*','',str(text))
text=re.sub(r'\\n','',str(text))
text = re.sub(r'[^\w\s]','',str(text)) #remove punctuation
text = re.sub(r'b', '', str(text))
text = re.sub(r'https\S+', '', text)
text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

f = open("process.txt", "w")
f.write(str(text))


#removing one column

keep_col = ['created_at','screen_name']
new_f = data[keep_col]
new_f.to_csv("update.csv", index=False)
update_data=pd.read_csv('update.csv')


import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO



TESTDATA = StringIO(text)


    
colname=['text']
data_text = pd.read_csv(TESTDATA,names=colname)
result = pd.concat([update_data, data_text], axis=1)
result.to_csv('process_data.csv',index=False)
'''
data_text['text'] = data_text['text'].str.strip()
data_text['index']=data_text.index
documents = data_text
#print(documents[:5])


stemmer=PorterStemmer()
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_docs = documents['text'].map(preprocess)
print(processed_docs[:10])


f = open("process.txt", "w")
f.write(str(text))





dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    #print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[2628])






#LDA
print('topic modeling result using LDA')
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


print("the doc to test")
print(processed_docs[3408])
for index, score in sorted(lda_model[bow_corpus[2628]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))   

'''

