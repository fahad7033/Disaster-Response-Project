import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages dataset and categories dataset,
    The output is a merged file of the two datasets   
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])
    
    return df



def clean_data(df_):
    '''
    1. Split categories into separate category columns.
    2. Convert category values to just numbers 0 or 1.
    3. Replace categories column in df with new category columns.
    4. Remove duplicates.

    '''
    
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';',expand=True))
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').apply(lambda x:x[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for col in categories:
        # set each value to be the last character of the string
        categories[col] = categories[col].astype(str).str.split('-').apply(lambda x:x[1])
        # convert column from string to numeric
        categories[col] = categories[col].astype(int)
        
    # drop the original categories column from `df`
    df_ = df_.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df_ = pd.concat([df_,categories],axis=1)
    # drop duplicates
    df_ = df_.drop_duplicates()
    
    return df_


def save_data(df_, database_filename):
    '''
    convert the data into database
    
    Parameters:
        df: input data
        database_filename: database file name
    '''
    # Create database engine
    engine = create_engine('sqlite:///'+database_filename)
    # Save df to database
    df_.to_sql('disaster_response', engine, index=False)
 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
