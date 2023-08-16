import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
class KeyWords:
  def __init__(self,stopwords,text):
    self.stopwords = stopwords
    self.text = text
    self.preprocessing_text = [KeyWords.pre_process(self,self.text)]
  def pre_process(self,text):
    text= text.lower()
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    text = [word for word in text if word not in self.stopwords]
    text = [word for word in text if len(word) >= 3]
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]
    return ' '.join(text)
  def train(self,train_mode = 'TF-IDF'):
    self.train_mode = train_mode
    if train_mode == 'TF-IDF':
      self.tfidf_vectorizer = TfidfVectorizer()
      self.Text_Tf_Idf = self.tfidf_vectorizer.fit_transform(self.preprocessing_text)
      self.Text_Tf_Idf_array = self.Text_Tf_Idf.toarray()[0]
  def max_fetures(self,n_features=10):
    if self.train_mode == 'TF-IDF':
      Text_Tf_Idf_array2 = self.Text_Tf_Idf_array.copy()
      words = self.tfidf_vectorizer.get_feature_names_out()
      words_max = []
      for i in range(n_features):
        index = np.argmax(Text_Tf_Idf_array2)
        word = words[index]
        Text_Tf_Idf_array2[index] = 0
        words_max.append(word)
      return words_max

Text = open(r"C:\Users\2022\Desktop\test\NLP-main\NLP-Test\ai.txt").read()
stop_words = set(stopwords.words('english'))
keyword = KeyWords(stop_words,Text)
keyword.train(train_mode = 'TF-IDF')
print(keyword.max_fetures(n_features=10))