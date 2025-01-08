from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words=stopwords.words('english')
import os.path
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

import os
import glob
import random
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
class Summarizer:
    
    def _init_(self):
        return;
    
    
    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = sent1.split(" ")
        sent2= sent2.split(" ")
        
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return (1 - cosine_distance(vector1, vector2))
    
    
    def build_similarity_matrix(self, sentences, stop_words):

        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):

            for idx2 in range(len(sentences)):

                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix
    
    
    def generate_summary(self):

        fileobj = open(r"C:\Users\Tanmayee\Desktop\bTECH_Project\Final Year Project App\sample.txt","r")
        sentences = fileobj.readlines()
        conversation = sentences[0].split(".")
        conversation.pop()

        top_n = len(conversation)//2
        stop_words = stopwords.words('english')
        print(conversation)
        summarize_text = []
        summary_ex = ""
        sentence_similarity_martix = self.build_similarity_matrix(conversation,stop_words)

        # Step 3 - Rank sentences in similarity martix

        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)

        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences

        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(conversation)), reverse=True)    

        print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):

          summarize_text.append("".join(ranked_sentence[i][1]))
          summary_ex = summary_ex + ranked_sentence[i][1]+"."

        # Step 5 - Offcourse, output the summarize texr
       
        print("\n\n\nSummarize Text: \n", ".".join(summarize_text))
        return summary_ex

class Classifier:
    def _init_(self):
        return;
    def classify(self,testSummary):
        
       
        clf_filename = 'classifier.pkl'
        cnt_vct_filename='countVectorizer.pkl'
        tfidf_trans_filename='tfidfTransformer.pkl'
        clf_pkl = open(clf_filename, 'rb')
        clf = pickle.load(clf_pkl)
        cntVectorizer_pkl=open(cnt_vct_filename, 'rb')
        countVectorizer=pickle.load(cntVectorizer_pkl)
        tfidf_trans_pkl=open(tfidf_trans_filename, 'rb')
        tfidfTransformer=pickle.load(tfidf_trans_pkl)
        
        test_vectorizer = countVectorizer.transform(testSummary).toarray()

        test_tfidf = tfidfTransformer.transform(test_vectorizer).toarray()

        predicted =clf.predict(test_tfidf)
       
        pre_arr=[]
        pre_arr.append(predicted)
      
        pred_pro = clf.predict_proba(test_tfidf)
        pre_arr.append(pred_pro)
        return pre_arr
        
class Report:
    def _init_(self):
        return;
    def generateAllReports(self):
        arrHELP=self.generateReport(os.getcwd() + '\ReportDataset\HELP')
        print(arrHELP)
        if(len(arrHELP)>=10):
            self.createBarGraph(arrHELP,'HELP')
        arrCOMPLAINT=self.generateReport(os.getcwd() + '\ReportDataset\COMPLAINT')
        if(len(arrCOMPLAINT)>=10):
            self.createBarGraph(arrCOMPLAINT,'COMPLAINT')
        arrFAKE=self.generateReport(os.getcwd() + '\ReportDataset\FAKE')
        if(len(arrFAKE)>=10):
            self.createBarGraph(arrFAKE,'FAKE')
       
        
        
    def createBarGraph(self,arrCat,categoryName):
        if(len(arrCat)>20):
            arrCat=arrCat[:20]
        
        x_labels = [val[1] for val in arrCat]
        y_labels = [val[0] for val in arrCat]
        plt.figure(figsize=(12, 3))
        ax = pd.Series(y_labels).plot(kind='bar')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        plt.xlabel('Words')
        plt.ylabel('Occurrence')
        plt.title(categoryName+' GRAPH')
		#gname=categoryName+'GRAPH.png';
		#gdir=os.path.join(os.getcwd(),static')
        plt.savefig( os.path.join(os.getcwd(),'static', categoryName+'GRAPH.png'),dpi=100,transparent=True,bbox_inches='tight')
		

    def generateReport(self,folder_path):
        arr = os.listdir(folder_path)
        print(len(arr))
        if(len(arr)>=2):
            
            file_list = glob.glob(os.path.join(folder_path, "*.txt"))

            corpus = []

            for file_path in file_list:
                with open(file_path) as f_input:
                    corpus.append(f_input.read())
            

            porter=PorterStemmer()
            print(corpus[0])
            print("Stemmed sentence")
            index=0;
            for i in corpus:
                
                corpus[index]=self.stemSentence(porter,i)
                index=index+1
            print("Corpus\t:\t",corpus)

            #Count Vectorizer
            vectorizer = CountVectorizer(max_features=1500, min_df=2, stop_words=stopwords.words('english'))
            X_vectorizer = vectorizer.fit_transform(corpus).toarray()
            print(X_vectorizer)
            print('Number of unique words chosen : ',vectorizer.get_feature_names())
            featureNames=vectorizer.get_feature_names()
            numeratorArr=self.calculate_numerator(X_vectorizer,len(vectorizer.get_feature_names()))
            myInt=len(arr)
            numeratorArr[:] = [x / myInt for x in numeratorArr]
            print(numeratorArr)
            numerator_merged_list = [(numeratorArr[i], featureNames[i]) for i in range(0, len(numeratorArr))] 
            print(numerator_merged_list)
            numerator_merged_list=self.sort(numerator_merged_list)
            print(numerator_merged_list)
            return numerator_merged_list
        else:
            print ('Cannot generate Report!!!')
            return [];
        
        return arr
    def stemSentence(self,porter,sentence):
        token_words=word_tokenize(sentence)
        token_words
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)
    def sort(self,tuples):
        return sorted(tuples, key = self.last,reverse=True) 
    
    def last(self,n):
        return n[0]  
    
    def calculate_numerator(self,vectorizer,size_of_features):
        numeratorArr=[]
        j=0
        while True:
            count=0
            for i in vectorizer:
                if(i[j]>0):
                    count=count+1
            numeratorArr.append(count)
            j=j+1
            if(j==size_of_features):
                break
            
        print(numeratorArr)
        return numeratorArr
        


from flask import Flask, request,render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/home1')
def home1():
    return render_template('home1.html')
@app.route('/reportNew')
def reportNew():
    return render_template('reportNew.html')

@app.route('/classify',methods=['POST'])
def classify():
    if request.method == 'POST':
        f = request.files["file_name"]
        f.save(f.filename)
        print('Filename : '+ f.filename)
        os.rename(f.filename,'sample.txt')
        #get extractive summary of the document
        summary=summarize()

        #using pickle, predict the probabilities of four classes
       
        Categories=['FAQ','Complaint','Training','Fake']
        fSummary=""
        for i in summary:
            fSummary=fSummary+i;
        fList=[];
        fList.append(fSummary)
        classifier=Classifier()
        probabilities=classifier.classify(fList)

        print ('Probabilities :',probabilities[1][0])
        cat_filename='categoryNames.pkl'
        cat_pkl=open(cat_filename,'rb')
        cat=pickle.load(cat_pkl)
        predictedCat=cat[probabilities[0][0]]
        print ("Categories : ",cat)
        print("Category Predicted : ",predictedCat)
        probabilities=probabilities[1][0]
		
        fileIndex='fileIndex.pickle'
        fileIndex_pkl=open(fileIndex,'rb')
        fileIndexArr=pickle.load(fileIndex_pkl)
        print(fileIndexArr)
        index=0;
        if predictedCat== 'HELP':
            save_path = os.getcwd() + '\ReportDataset\HELP'
            index=fileIndexArr[0]+1;
            fileIndexArr[0]=index;
        elif predictedCat == 'FAKE':
            save_path = os.getcwd() + '\ReportDataset\FAKE'
            index=fileIndexArr[1]+1;
            fileIndexArr[1]=index;
        elif predictedCat == 'COMPLAINT': 
            save_path = os.getcwd() + '\ReportDataset\COMPLAINT'
            index=fileIndexArr[1]+1;
            fileIndexArr[2]=index;
        with open('fileIndex.pickle', 'wb') as b:
            pickle.dump(fileIndexArr,b)
        name_of_file = "Summary"+ str(index)
        completeName = os.path.join(save_path , name_of_file+".txt")
        file1 = open(completeName , "w")
        print (fSummary)
        file1.write(fSummary)
        file1.close()
        os.remove('sample.txt')
        return render_template('Result.html',probability=probabilities,categories=cat,extractive_summary=format(summary),resultCat=predictedCat)

@app.route('/report',methods=['POST'])
def report():
    W1=['AmazonPay','Return','Earphones','Payment']
    F1=[20,15,7,18]
    W2=['Flight','Delvery','Fraud','Refund']
    F2=[30,20,5,9]
    report=Report()
    arr=report.generateAllReports()
    return render_template('reportNew.html');
def summarize():
    summary="";
    summarizer=Summarizer()
    summary=summarizer.generate_summary()
    return summary

if __name__ == "__main__":
    app.run(debug=True)
	

"""
class ExtractiveSummarizer:
    
    def _init_(self):
        return
    
    
    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = sent1.split(" ")
        sent2= sent2.split(" ")
        
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return (1 - cosine_distance(vector1, vector2))
    
    
    def build_similarity_matrix(self, sentences, stop_words):

        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):

            for idx2 in range(len(sentences)):

                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix
    
    
    def generate_summary(self, sentences):

        # fileobj = open(r"C:\Users\Tanmayee\Desktop\bTECH_Project\Final Year Project App\sample.txt","r")
        # sentences = fileobj.readlines()
        conversation = sentences.split(".")
        conversation.pop()

        top_n = len(conversation)//2
        stop_words = stopwords.words('english')
        print(conversation)
        summarize_text = []
        summary_ex = ""
        sentence_similarity_martix = self.build_similarity_matrix(conversation,stop_words)

        # Step 3 - Rank sentences in similarity martix

        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)

        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences

        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(conversation)), reverse=True)    

        print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):

          summarize_text.append("".join(ranked_sentence[i][1]))
          summary_ex = summary_ex + ranked_sentence[i][1]+"."

        # Step 5 - Offcourse, output the summarize texr
       
        print("\n\n\nSummarize Text: \n", ".".join(summarize_text))
        return summary_ex
    
class AbstractiveSummarizer:

    def _init_(self):
        return
    
    def generate_summary(self, sentences):
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        inputs = tokenizer.encode("summarize: " + sentences, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(summary)
        return summary
"""