 # Place your imports here
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords

class TextRetrieval():

  #For preprocessing
  punctuations = ""
  stop_words=set()

  #For VSM definition
  vocab = np.zeros(200)
  dataset = None
  K = 3 #

  def __init__(self):
    ##
    #TODO: obtain the file "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    # and store it locally in a location accessible directly by this script (e.g. same directory don't use absolute paths)
    
    ### TODO: Initialize punctuations (a string) and stop_words (a set)
    # Define self.punctuations to be any '"\,<>./?@#$%^&*_~/!()-[]{};:
    # Define self.stop_words from stopwords.words('english')


  def read_and_preprocess_Data_File(self):
    ### Reads the test.csv file and iterates over every document content (entry in the column 2)
    ### removes leading and trailing spaces, transforms to lower case, remove punctuation, and removes stopwords
    ### Stores the formated information in the same "dataset" object

    dataset = pd.read_csv("test.csv",header=None)
    punctuations = self.punctuations
    stop_words = self.stop_words

    dataset.head()
    for index, row in dataset.iterrows():
      line = row[2]
      #TODO: Implement removing stopwords and punctuation

      dataset.loc[index, 2] = ' '.join(words)
      
    self.dataset = dataset #Set dataset as object attribute


  #### Bit Vector with Dot Product

  def build_vocabulary(self): #,collection):
    ### Return an array of 200 most frequent works in the collection
    ### dataset has to be read before calling the vocabulary construction

    #TODO: Create a vocabulary. Assume self.dataset has been preprocessed. Count the ocurrance of the words in the dataset. Select the 200 most common words as your vocabulary vocab. 
    self.vocab = vocab

  def text2BitVector(self,text):
    ### return the bit vector representation of the text

    #TODO: Use self.vocab (assume self.vocab is created already) to transform the content of text into a bitVector
    #Use the order in the vocabulary to match the order in the bitVector
    
    return bitVector

  def bit_vector_score(self, query,doc):
    ### query and doc are the space-sparated list of words in the query and document
    q = self.text2BitVector(query)
    d = self.text2BitVector(doc)

    #TODO: compute the relevance using q and d

    return relevance

  def adapt_vocab_query(self,query):
    ### Updates the vocabulary to add the words in the query

    #TODO: Use self.vocab and check whether the words in query are included in the vocabulary
    #If a word is not present, add it to the vocabulary (new size of vocabulary = original + #of words not in the vocabulary)
    #you can use a local variable vocab to work your changes and then update self.vocab
      
    self.vocab = vocab

  def execute_search_BitVec(self,query):
    ### executes the computation of the relevance score for each document
    ### but first it verifies the query words are in the vocabulary
    #e.g.: query = "olympic gold athens"


    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query

    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using bit_vector_score
    
    return relevances # in the same order of the documents in the dataset

  #### TF-IDF with Dot Product

  def compute_IDF(self,M,collection):
    ### M number of documents in the collection; collection: documents (i.e., column 3 (index 2) in the dataset)

    #To solve this question you should use self.vocab

    self.IDF  = np.zeros(self.vocab.size) #Initialize the IDFs to zero
    #TODO: for word in vocab: Compute the IDF frequency of each word in the vocabulary using math.log


  def text2TFIDF(self,text, applyBM25_and_IDF=False):
    ### returns the bit vector representation of the text

    #TODO: Use self.vocab, self.K and self.IDF to compute the TF-IDF representation of the text

    tfidfVector = np.zeros(vocab.size)

    for word in vocab:
      if word in text.split():
        #TODO: Set the value of TF-IDF to be (temporarily) equal to the word count of word in the text
        if applyBM25_and_IDF:
            #TODO: update the value of the tfidfVector entry to be equal to BM-25 (of the word in the document) multiplied times the IDF of the word
    return tfidfVector

  #grade (enter your code in this cell - DO NOT DELETE THIS LINE)
  def tfidf_score(self,query,doc, applyBM25_and_IDF=False):
    q = self.text2TFIDF(query)
    d = self.text2TFIDF(doc,applyBM25_and_IDF)

    #TODO: compute the relevance using q and d

    return relevance

  def execute_search_TF_IDF(self,query):
    #DIFF: Compute IDF
    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query
    # global IDF
    self.compute_IDF(self.dataset.shape[0],self.dataset[2]) #IDF is needed for TF-IDF and can be precomputed for all words in the vocabulary and a given fixed collection (this excercise)

    #For this function, you can use self.IDF and self.dataset
    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using tfidf_score
    
    return relevances # in the same order of the documents in the dataset




if __name__ == '__main__':
    tr = TextRetrieval()
    tr.read_and_preprocess_Data_File() #builds the collection
    tr.build_vocabulary()#builds an initial vocabulary based on common words
    queries = ["olympic gold athens", "reuters stocks friday", "investment market prices"]
    print("#########\n")
    print("Results for BitVector")
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_BitVec(query)
      #TODO: Once the relevances are computed, print the top 5 most relevant documents and the bottom 5 least relevant (for your reference) 
      

    print("#########\n")
    print("Results for TF-IDF")
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_TF_IDF(query)
      #TODO: Once the relevances are computed, print the top 5 most relevant documents and the bottom 5 least relevant (for your reference) 
