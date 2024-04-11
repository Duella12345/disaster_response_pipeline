# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath = "DisasterResponse.db"):
    """Open sqlite database and load table as dataframe. combine message and genre columns for X variable and 
    output the encoded category columns as Y.

    Parameters:
    database_filepath -- string database name to save the dataframe to (default "DisasterResponse.db")

    Returns:
    Dataframes: X,Y
    """
    # define features and label arrays
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine) 
    X = df['message'] + " " + df['genre']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y


def tokenize(text):
    """tokenize string by words, remove stopwords, lemmatize and output as list

    Parameters:
    text -- string to tokenize

    Returns:
    list: clean_tokens
    """
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def display_results(y_test, y_pred):
    """take test y and pred y and test for and print precision, recall and f1-score

    Parameters:
    y_test -- array of test y to test against
    y_pred -- array of y prections to test
    """
    
    target_names = y_test.columns
    
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """take text and output starting verb

        Parameters:
        text -- string to extract starting verb

        Returns:
        boolean: whether verb present or not
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
class Genre(BaseEstimator, TransformerMixin):

    def genre(self, text):
        """take text and outputs genre

        Parameters:
        text -- string to extract genre

        Returns:
        string: genre string
        """
        return text.split()[-1]

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """take X dataframe and outputs genres as dummies columns

        Parameters:
        X -- dataframe to extract genre

        Returns:
        Dataframe: genre dataframe
        """
        X_tagged = pd.get_dummies(pd.Series(X).apply(self.genre))
        return pd.DataFrame(X_tagged)


def build_model():
    """creates pipeline with TFIDF vectorizer combined with StartingVerbExtractor, Genre and using a MultiOutputClassifier
    to run LogisticRegression on each category. The model pipeline is then instantiated  with GridSearchCV.

    Returns:
    Pipeline: model_pipeline 
    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))),
            ])),

            ('starting_verb', StartingVerbExtractor()),
            # ('genre', Genre())
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=500, solver='lbfgs'))),
    ])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__penalty': ['l2', None],
    }


    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline

def evaluate_model(model, X_test, Y_test):
    """takes MultiOutputClassifier model, X_test, Y_test lists, predicts y_pred from X_test 
    and prints precision, recall and f1-score using display_results

    Parameters:
    model -- MultiOutputClassifier model
    X_test -- X test data (model input) list
    Y_test -- Y test data (model input) list
    """

    y_pred = model.predict(X_test)
    display_results(Y_test, y_pred)


def save_model(model, model_filepath = "classifier.pkl"):
    """save model as a pkl file

    Parameters:
    model -- MultiOutputClassifier model
    model_filepath -- file path for model to be saved to (default "classifier.pkl")
    """
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if (len(sys.argv) == 1):
        print('Loading data...\n    DATABASE: {}'.format("DisasterResponse.db"))
        X, Y = load_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format("classifier.pkl"))
        save_model(model)

        print('Trained model saved!')
    elif (len(sys.argv) == 3):
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print(str(len(sys.argv)) + 'Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()