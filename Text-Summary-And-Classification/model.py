from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pandas as pd
import nltk
nltk.download('stopwords')
stop_words=stopwords.words('english')
import os.path
import pickle
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class Summarizer:
    

    def _init_(self):
        return;
    
    def get_summary_all(self):
        df=pd.read_csv('amazonnewdata.csv', encoding = 'cp1252')
        print (df);
        gd=df.groupby('chat_id')
        chatid = list(gd.groups.keys())
        gd=df.groupby('chat_id')
        chatid = list(gd.groups.keys())
        j=0
        for i in range(len(gd)):

            df1= pd.DataFrame
            df1 = gd.get_group(chatid[i])
            new_text_df1=pd.DataFrame
            new_text_df1=df1['Conversation']
            print(new_text_df1)
            print("length of new_text_df is",len(new_text_df1))
            sentence = self.retrieve_Sentences(new_text_df1, j )
            for ind in df1.index:
                    vector=""
                    sent= df1['Conversation'][ind]
                    str1=sent.replace("[^a-zA-Z]", " ").split(" ")
                    for word in str1:
                        if word not in stop_words:
                            vector=vector+word+" "
                    #print(vector)
            
            extract_summary = self.generate_summary(j,chatid[i], new_text_df1,2 )

            extractive_summary = '.'.join(map(str , extract_summary))

            if df1['Category'][j]== 'HELP':
                save_path = os.getcwd() + '\Dataset\HELP'
            elif df1['Category'][j] == 'FAKE':
                save_path = os.getcwd() + '\Dataset\FAKE'
            elif df1['Category'][j] == 'COMPLAINT': 
                save_path = os.getcwd() + '\Dataset\COMPLAINT'

            name_of_file = "Summary"+ str(chatid[i])
            completeName = os.path.join(save_path , name_of_file+".txt")
            file1 = open(completeName , "w")
            file1.write(extractive_summary)
            file1.close()
            j = j+len(new_text_df1)
            print("j:",j)
    def sentence_similarity(self,sent1, sent2,stopwords=None):
        if stopwords is None:
            stopwords = []

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

        return 1 - cosine_distance(vector1, vector2)
		
    def build_similarity_matrix(self,sentences, stop_words):

        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):

            for idx2 in range(len(sentences)):

                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix
    def generate_summary(self,j , chatid , df1, top_n):

        stop_words = stopwords.words('english')

        summarize_text = []

        # Step 1 - Read text anc split it

        #sentences =  read_article(file_name)

        sentences=[]
        sentences = self.retrieve_Sentences(df1, j)
        # Step 2 - Generate Similary Martix across sentences
        print(sentences)
        sentence_similarity_martix = self.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix

        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)

        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences

        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    

        print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):

          summarize_text.append("".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        print("The summary of chat id ",chatid, " is")
        print("\n\n\nSummarize Text: \n", ".".join(summarize_text))
        return summarize_text
    def retrieve_Sentences(self,df, n):
        sentences1=[]
        print("length of df in func is:",len(df))
        print("the indices of this df are")
        print(df.index)
        j=n
        for j in df.index:
            sentences1.append(df[j])
        return sentences1
	
class Classifier:
    
    def classify(self):
        conversation_data = load_files(r"C:\Users\Anamika\Desktop\Final Year Programs\Dataset",encoding = 'cp1252')
        X_full, y_full = conversation_data.data, conversation_data.target
        category=conversation_data.target_names
        #printing all categories
        print("Categories : ",len(category))
        print('-----------------------------------------')
        for cat in category:
            print(cat)
        print('-----------------------------------------')
        print('Length of convesation array : ',len(X_full))
        print('Length of label array: ',len(y_full))
        print('-----------------------------------------')
        print('First Converastion : ')
        sampleX=X_full[0]
        sampleY=y_full[0]
        print(sampleX)
        print('\nLabel of first Conversation : ')
        print(sampleY," , that is category is - ",category[sampleY])
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=0)
        print('Length of training data : ',len(X_train))
        print('Training data : ',X_train)
        print('Length of labels of training data',len(y_train))
        print('labels of training data : ',y_train)
        print('Length of testing data : ',len(X_test))
        print('Length of labels of testing data',len(y_test))
        vectorizer = CountVectorizer(max_features=1500, min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
        X_vectorizer = vectorizer.fit_transform(X_train).toarray()
        print('Count vectorizer : ',X_vectorizer)
        print('Number of unique words chosen : ',len(vectorizer.get_feature_names()))
        print('Unique words : ',vectorizer.get_feature_names())
        print('First document vector : ',X_vectorizer[0])
        print('Length : ',len(X_vectorizer[0]))
        tfidfconverter = TfidfTransformer()
        X_tfidf = tfidfconverter.fit_transform(X_vectorizer).toarray()
        print('tfidf of first document : ')
        print(X_tfidf[0])
        print('Length : ',len(X_tfidf[0]))
        print('Length of whole tfidf : ')
        print(len(X_tfidf))
        text_clf = MultinomialNB().fit(X_tfidf, y_train)
        X_test_vectorizer = vectorizer.transform(X_test).toarray()

        X_test_tfidf = tfidfconverter.transform(X_test_vectorizer).toarray()
        

        predicted = text_clf.predict(X_test_tfidf)
       
        #print (predicted)
        #print (y_test)
        print("Accuracy Score : ",accuracy_score(y_test,predicted))
        # Dump the trained decision tree classifier with Pickle
        pkl_filename = 'classifier.pkl'
        pkl2_fname='countVectorizer.pkl'
        pkl3_fname='tfidfTransformer.pkl'
        pkl4_fname='categoryNames.pkl'
        # Open the file to save as pkl file
        pkl = open(pkl_filename, 'wb')
        pkl2=open(pkl2_fname,'wb')
        pkl3=open(pkl3_fname,'wb') 
        pkl4=open(pkl4_fname,'wb')
        
        pickle.dump(text_clf, pkl)
        #print(pkl)
        pickle.dump(vectorizer, pkl2)
        pickle.dump(tfidfconverter, pkl3)
        pickle.dump(category,pkl4)
        # Close the pickle instances
        pkl.close()
        pkl2.close()
        pkl3.close()
        pkl4.close()
        print ("Classifier Pickled\t:\n",text_clf)
	
summarizer=Summarizer()
summarizer.get_summary_all()
classifier=Classifier()
classifier.classify()