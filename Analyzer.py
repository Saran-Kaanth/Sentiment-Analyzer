from google.protobuf.symbol_database import Default       #importing required libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem.porter import PorterStemmer

tokenizer=RegexpTokenizer(r'\w+')
ps=PorterStemmer()

stop_words_list= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all','product']

with open("models\models.p","rb") as file:
    model=pickle.load(file)
    vectorizer=model["vectorizer"]
    nb_classifier=model["model"]

def cleaned_text(text):          #Preprocessing like tokenize,stemming,removing stopwords
    text=text.lower()
    
    tokens=tokenizer.tokenize(text)
    stop_word=[token for token in tokens if token not in stop_words_list]
    
    stemmed_tokens=[ps.stem(token) for token in stop_word]
    
    clean_text=" ".join(stemmed_tokens)
    
    return clean_text

st.set_page_config(layout="wide")
st.empty()
st.title("Sentiment Analyzer")
col1,col2=st.columns(2)
review_inp=st.sidebar.text_input("Enter your review","")
filename=st.sidebar.file_uploader("Pick a file",type=("txt","csv"))
st.sidebar.warning("Note: Please Don't use the two options at the same time. To move to another make sure to clear the another.")


if filename is not None:                    #for csv files consists of many reviews
    df=pd.read_csv(filename)
    review_text=[]
    for col in df.columns:
        if col.lower()=="review" or col.lower()=="reviews":
            for val in df[col]:
                review_text.append(val)
    
    arr_clean=np.array([cleaned_text(i) for i in review_text])
    arr_vec=vectorizer.transform(arr_clean).toarray()
    predicted_values=nb_classifier.predict(arr_vec)

    unique,frequency=np.unique(predicted_values,return_counts=True)
    senti_dict={"Negative":frequency[0],"Positive":frequency[1]}

    senti_df=pd.DataFrame(frequency,index=[unique[0],unique[1]],columns=["Frequency of Sentiments"])

    st.bar_chart(senti_df)
    st.title("Thanks for Your Time!!!")
        
elif review_inp=="":                               #default message
    st.title("Welcome To Review Finder")

elif review_inp is not None:                    #For single line of text
    st.write("Your review was: ","'",review_inp,"'")
    review=np.array([review_inp])
    vectorize=vectorizer.transform(np.array([cleaned_text(i) for i in review])).toarray()
    predict_sentiment=nb_classifier.predict(vectorize)[0]
    if predict_sentiment.lower()=="positive":
        st.title("Hooyahh!! You got an Positive Review")
    elif predict_sentiment.lower()=="negative":
        st.title("Oops!! It was a Negative Review! Next Time!!")






