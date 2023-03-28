import json
import time
import pickle
from sklearn.datasets import load_files

from nltk.stem import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from flask import jsonify
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

###### INIT CONSTANT
SEVERITY = ["Critical", "High", "Low", "Medium"]
TYPE = ["API", "Configuration/ DevOps", "Database", "UI", "UI logic"]

# load model
with open('data/bug_severity/bug_severity', 'rb') as severity_model:
    bug_severity_model = pickle.load(severity_model)

with open('data/bug_type/bug_type', 'rb') as type_model:
    bug_type_model = pickle.load(type_model)

# load data
bug_severity_data = load_files(r"data\bug_severity\train")
bug_type_data = load_files(r"data\bug_type\train")

############################### 

@app.route('/')
def hello():
    res = 'Hello, World!'

    return res

@app.route('/bug-severity-prediction', methods=['POST'])
def bug_severity_prediction():  
    record = json.loads(request.data)   
    SEARCH_TERM = record['search_term'] 

    X, y = bug_severity_data.data, bug_severity_data.target
    X[0] = SEARCH_TERM

    START_TIME = time.time()

    stemmer = WordNetLemmatizer()
# Text Preprocessing
    documents = []
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        # Converting to Lowercase
        document = document.lower()
        
        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        
        documents.append(document)

    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(documents).toarray()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0, train_size=None, shuffle=False)

# Prediction
    y_pred2 = bug_severity_model.predict(X)

    END_TIME = time.time()

    elapsed_time = END_TIME - START_TIME
    
    res = jsonify({
                "bug_description": SEARCH_TERM,
                "result": SEVERITY[y_pred2[0]]
            })

    return res

@app.route('/bug-type-prediction', methods=['POST'])
def bug_type_prediction():  
    record = json.loads(request.data)   
    SEARCH_TERM = record['search_term'] 

    X, y = bug_type_data.data, bug_type_data.target
    X[len(X)-1] = SEARCH_TERM
    # X.append(SEARCH_TERM)

    START_TIME = time.time()

    stemmer = WordNetLemmatizer()
# Text Preprocessing
    documents = []
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        # Converting to Lowercase
        document = document.lower()
        
        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        
        documents.append(document)

    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(documents).toarray()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0, train_size=None, shuffle=False)

# Prediction
    y_pred = bug_type_model.predict(X)

    END_TIME = time.time()

    elapsed_time = END_TIME - START_TIME
    
    res = jsonify({
                "bug_description": SEARCH_TERM,
                "result": TYPE[y_pred[len(X)-1]]
            })

    return res

if __name__=="__main__":
    app.run("0.0.0.0")