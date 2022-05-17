import sys
import pandas as pd
import numpy as np
import nltk
import re
from sqlalchemy import create_engine
import pickle
import random
import warnings
warnings.filterwarnings('ignore') 

nltk.download(['words', 'punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'wordnet'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    '''
    Args:
    
    database_filepath - the database filepath used for the model: sqlite:///Disaster_data.db  
    
    Returns:
 
    Y - the dependent variables 
    X - dataframe of independent variables used to predict Y
    categories_names - independent variables names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse, con=engine) #database_filepath
    X = df['message']
    Y = df.iloc[:,4:]
    categories_names = Y.columns
    return X, Y, categories_names


def tokenize(text):
    '''
    Args: 
    text - the message column
    
    Returns:
    clean_tokens - return the input text clean and tokenized
    '''
    #check for urls in the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text) 
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    #tokenize the text 
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    #lemmatize the text
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''
    Returns:
    cv - the new model tunned with GridSearchCV
    '''
    # creating a machine learning pipeline that uses Random Forest method to classify the messages
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # setting the parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names =Y.columns))
    
    accuracy = (y_pred == y_test).mean()
    print("The calculated accuracy is :",accuracy)
    


def save_model(model, model_filepath):
    #source:Grepper.com
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()