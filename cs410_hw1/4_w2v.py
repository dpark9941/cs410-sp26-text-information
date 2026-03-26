import numpy as np
import numpy.linalg as la
import nltk
import pandas as pd
from nltk.corpus import stopwords


##TODO: Send a private message to instructor on Campuswire to note which additional libraries you are using
import gensim
import gensim.downloader as api
from collections import Counter
import re
import math


class TextRetrieval():

  #For preprocessing
  punctuations = ""
  stop_words=set()

  # For Word2vec definition
  w2v_model = None

  #For VSM definition
  vocab = np.zeros(200)
  dataset = None
  K = 3 # total number of docs containing W in the collection (Doc frequency))

  def __init__(self):
    ## 
    #TODO: obtain the file "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    # and store it locally in a location accessible directly by this script (e.g. same directory don't use absolute paths)
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    filename = "test.csv"
    pd.read_csv(url, header=None).to_csv(filename, index=False, header=False)
    print("File downloaded successfully")

    ### TODO: Initialize punctuations (a string) and stop_words (a set)
    # Define self.punctuations to be any '"\,<>./?@#$%^&*_~/!()-[]{};:
    # Define self.stop_words from stopwords.words('english')
    self.punctuations = '\'\"\\,<>./?@#$%^&*_~/!()-[]{};:'
    self.stop_words = set(stopwords.words('english'))

    # Initialize pretrained word2vec model
    try:
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    except FileNotFoundError:
        print("Local file not found, downloading via gensim-data...")
        self.w2v_model = api.load('word2vec-google-news-300')

  def read_and_preprocess_Data_File(self):
    ### Reads the test.csv file and iterates over every document content (entry in the column 2)
    ### removes leading and trailing spaces, transforms to lower case, remove punctuation, and removes stopwords
    ### Stores the formated information in the same "dataset" object

    dataset = pd.read_csv("test.csv",header=None)
    punctuations = self.punctuations
    stop_words = self.stop_words

    for index, row in dataset.iterrows():
      line = row[2]
      #TODO: Implement removing stopwords and punctuation
      line = re.sub(r'<[^>]+>', ' ', line) #Remove HTML tags using regex
      line = re.sub(r'\d+', '', line) #Remove numbers using regex
      for p in punctuations:
          line = line.replace(p, " ")
      words = line.split()
      words = [word.lower() for word in words]
      words = [word for word in words if word not in stop_words]

      dataset.loc[index, 2] = ' '.join(words)
      
    self.dataset = dataset #Set dataset as object attribute
    dataset.to_csv("preprocessed_test.csv", index=False, header=False) #Save the preprocessed dataset in a new file for your reference


  #### Bit Vector with Dot Product

  def build_vocabulary(self): #,collection):
    ### Return an array of 200 most frequent works in the collection
    ### dataset has to be read before calling the vocabulary construction

    #TODO: Create a vocabulary. Assume self.dataset has been preprocessed.
    # Count the ocurrance of the words in the dataset. Select the 200 most common words as your vocabulary vocab. 
    all_words = []
    for index, row in self.dataset.iterrows():
      line = row[2]
      words = line.split()
      all_words.extend(words)
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(200)
    vocab = np.array([word for word, count in most_common_words])
    self.vocab = vocab
    # print(vocab.shape)
    # print(most_common_words[:10]) #Print the 10 most common words in the dataset for your reference

  def text2W2V(self,text):
    ### return the w2v representation of the text

    #TODO: Use self.vocab (assume self.vocab is created already) to transform the content of text into a bitVector
    #Use the order in the vocabulary to match the order in the bitVector
    w2v = np.zeros(300) # Initialize with Word2Vec dimension
    for word in text.split():
        if word in self.vocab:
            if word in self.w2v_model: # Check if word exists in w2v model
                w2v += self.w2v_model[word]
            
    return w2v

  def w2v_score(self, query,doc):
    ### query and doc are the space-sparated list of words in the query and document
    q = self.text2W2V(query)
    d = self.text2W2V(doc)

    #TODO: compute the relevance using q and d
    denominator = la.norm(q) * la.norm(d)
    if denominator == 0:
        relevance = 0.0
    else:
        relevance = np.dot(q, d) / denominator # Cosine similarity
    return relevance

  def adapt_vocab_query(self,query):
    ### Updates the vocabulary to add the words in the query
    #TODO: Use self.vocab and check whether the words in query are included in the vocabulary
    #If a word is not present, add it to the vocabulary (new size of vocabulary = original + #of words not in the vocabulary)
    #you can use a local variable vocab to work your changes and then update self.vocab
    vocab = self.vocab
    for word in query.split():
        if word not in vocab:
            vocab = np.append(vocab, word)
    self.vocab = vocab

  def execute_search_w2v(self,query):
    ### executes the computation of the relevance score for each document
    ### but first it verifies the query words are in the vocabulary
    #e.g.: query = "olympic gold athens"

    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query
    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using bit_vector_score
    for idx, doc in enumerate(self.dataset[2]): #Iterate over the documents in the dataset (column 3 (index 2))
        relevance = self.w2v_score(query, doc) 
        relevances[idx] = relevance
    # print(relevances.max())
    
    return relevances # in the same order of the documents in the dataset


if __name__ == '__main__':
    tr = TextRetrieval()
    tr.read_and_preprocess_Data_File() #builds the collection
    tr.build_vocabulary()#builds an initial vocabulary based on common words
    queries = ["olympic gold athens", "reuters stocks friday", "investment market prices"]
    print("#########\n")
    print("Results for W2V")
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_w2v(query)
      #TODO: Once the relevances are computed, print the top 5 most relevant documents and the bottom 5 least relevant (for your reference) 
      print("Top 5 most relevant documents:")
      top_5_indices = np.argsort(relevance_docs)[-5:][::-1] # Get indices of top 5 relevant documents
      for idx in top_5_indices:
          print(f"Document {idx}: {tr.dataset.iloc[idx, 1]} (Relevance Score: {relevance_docs[idx]})")
      print("Bottom 5 least relevant documents:")
      bottom_5_indices = np.argsort(relevance_docs)[:5] # Get indices of bottom 5 relevant documents
      for idx in bottom_5_indices:
          print(f"Document {idx}: {tr.dataset.iloc[idx, 1]} (Relevance Score: {relevance_docs[idx]})")
