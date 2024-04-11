# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath = "./data/messages.csv", categories_filepath = "./data/categories.csv"):
    """Load in train data from csv files.

    Parameters:
    messages_filepath -- message file path string (default "./data/messages.csv")
    categories_filepath -- category file path string (default "./data/categories.csv")

    Returns:
    dataframes: messages, categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)


    return messages, categories


def clean_data(messages, categories):
    """Clean up and merge the two data frames, remove constant columns and one hot encode categories.

    Parameters:
    messages -- message dataframe
    categories -- category dataframe

    Returns:
    dataframe: df
    """
     # clean data

    # merge datasets
    df = pd.merge(messages, categories, on="id")

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [name[0:-2] for name in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to 0 and 1s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace the one 2 with a 1
    categories = categories.replace(2,1)

    # drop column without any 1s
    categories = categories.drop(['child_alone'], axis=1)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(messages, categories, left_index=True, right_index=True)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename = "DisasterResponse.db"):
    """Create sqlite database and load dataframe as a table into new database.

    Parameters:
    df -- cleaned up dataframe
    database_filename -- string database name to save the dataframe to (default "DisasterResponse.db")
    """
    # load to database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("DisasterResponse", engine, index=False,  if_exists='replace')  


def main():
    if len(sys.argv) == 1:

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format("messages.csv", "categories.csv"))
        messages, categories = load_data()

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format("DisasterResponse.db"))
        save_data(df)
        
        print('Cleaned data saved to database!')
    elif len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
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