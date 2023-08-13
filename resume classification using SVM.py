# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:05:33 2023

@author: Zahid Hasan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from sklearn.model_selection import GridSearchCV
import joblib
nltk.download('stopwords')



# Replace backslashes with forward slashes
dataset_path = r'C:\Users\Zahid Hasan\Downloads\resume classification\Resume\Resume.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(dataset_path)

# Display the first few rows of the DataFrame
print(df.head())
# drop not using columns
df.drop(columns = ['ID', 'Resume_html'], inplace = True)
df

STEMMER = nltk.stem.porter.PorterStemmer()

def preprocess(txt):
    # convert all characters in the string to lower case
    txt = txt.lower()
    # remove non-english characters, punctuation and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    # tokenize word
    txt = nltk.tokenize.word_tokenize(txt)
    # remove stop words
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]
    # stemming
    txt = [STEMMER.stem(w) for w in txt]

    return ' '.join(txt)

df['Resume'] = df['Resume_str'].apply(lambda w: preprocess(w))
# drop original text column
df.pop('Resume_str')
df
df.info()
df['Category'].value_counts()
df['Category'].value_counts().sort_index().plot(kind='bar', figsize=(12, 6))
plt.show()

from matplotlib.gridspec import GridSpec
count=df['Category'].value_counts()
label=df['Category'].value_counts().keys()

plt.figure(1, figsize=(25,25))
grid=GridSpec(2,2)

cmap=plt.get_cmap('coolwarm')

color=[cmap(i) for i in np.linspace(0, 1, 5)]
plt.subplot(grid[0,1], aspect=1, title='Distribution')

pie=plt.pie(count, labels=label, autopct='%1.2f%%')
plt.show()

# create list of all categories
categories = np.sort(df['Category'].unique())
categories

# create new df for corpus and category
df_categories = [df[df['Category'] == category].loc[:, ['Resume', 'Category']] for category in categories]
df_categories[10]


# word frequency for each category
def wordfreq(df):
    count = df['Resume'].str.split(expand=True).stack().value_counts().reset_index()
    count.columns = ['Word', 'Frequency']

    return count.head(10)

fig = plt.figure(figsize=(32, 64))

for i, category in enumerate(np.sort(df['Category'].unique())):
    wf = wordfreq(df_categories[i])

    fig.add_subplot(12, 2, i + 1).set_title(category)
    plt.bar(wf['Word'], wf['Frequency'])
    plt.ylim(0, 3500)

plt.show()
plt.close()


nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords

#defining stopwords
stop_words = stopwords.words('english')

#function which stop words and the words less than 2 characters
def remove_stop_words (text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
      result.append(token)

  return result

df['clean'] = df['Resume'].apply(remove_stop_words).astype(str)

df['clean'][0]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df['clean'], df['Category'], test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer

# vectorize text data
vectorizer = CountVectorizer()
conuntvectorizer_train = vectorizer.fit_transform(X_train).astype(float)
conuntvectorizer_test = vectorizer.transform(X_test).astype(float)


# using grid search for find optimal values for RF classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RFC = RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

grid = GridSearchCV(estimator=RFC, param_grid=param_grid, cv= 5, scoring='accuracy', return_train_score=False, verbose=1)
grid_search = grid.fit(conuntvectorizer_train, Y_train)

print(grid_search.best_params_)

from sklearn.ensemble import RandomForestClassifier

RF_Model = RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=8, criterion='gini')
RF_Model.fit(conuntvectorizer_train, Y_train)


# Save the trained model to a file
model_filename = 'trained_model.pkl'
joblib.dump(RF_Model, model_filename)
prediction=RF_Model.predict(conuntvectorizer_test)
prediction
      
print("training Score: {:.2f}".format(RF_Model.score(conuntvectorizer_train, Y_train)))
print("test Score: {:.2f}".format(RF_Model.score(conuntvectorizer_test, Y_test)))

from sklearn import metrics
print("model report: %s: \n %s\n" % (RF_Model, metrics.classification_report(Y_test, prediction)))



# Calculate the confusion matrix
conf_matrix = metrics.confusion_matrix(Y_test, prediction)

# Plotting Confusion Matrix as Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Calculate precision, recall, and F1-score for each category
classification_rep = classification_report(Y_test, prediction, output_dict=True)
classification_metrics = pd.DataFrame(classification_rep).T

# Plotting Precision, Recall, and F1-Score for each category
plt.figure(figsize=(12, 6))
classification_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Precision, Recall, and F1-Score for Each Category")
plt.xlabel("Category")
plt.ylabel("Score")
plt.show()










































