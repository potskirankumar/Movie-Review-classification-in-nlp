import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy
from kcontractions import kkcontractions
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


df=pd.read_csv('movie.csv')


df['label'].value_counts()



df['text']=df['text'].apply(lambda x: x.lower())

exclude=string.punctuation

def remove_punc(text):
    for char in exclude:
        text=text.replace(char,'')
    return text

df['text']=df['text'].apply(lambda x: remove_punc(x))

from nltk.corpus import stopwords

def remove_stopw(text):
    return " ".join([word for word in str(text).split() if word not in stopword_l])


df['text']=df['text'].apply(lambda x: remove_stopw(x))

def tokens_l(text):
    tokens=nltk.word_tokenize(text)
    tokens=[token.strip() for token in tokens]
    return tokens
    
df['text']=df['text'].apply(lambda x: tokens_l(x))

wlm=WordNetLemmatizer()

def lemmatize_t(text):
    tokens=[wlm.lemmatize(t) for t in text]
    return " ".join(tokens)

df['text']=df['text'].apply(lemmatize_t)

tfidf=TfidfVectorizer()
X_tfidf=tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    Y,
    test_size=0.2,
    random_state=2002,
    stratify=df.label
)
clf=Pipeline([
    ('KNN',KNeighborsClassifier())
])
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))
