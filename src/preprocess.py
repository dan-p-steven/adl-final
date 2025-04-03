import pandas as pd
import spacy
from tqdm import tqdm
import time

import numpy as np

# Load Spacy's NLP model to use for in vector embeddings
# spacy_nlp = spacy.load('en_core_web_lg')

CHUNKSIZE = 50000

def generate_save_word_embeddings(dataset_read_path):
    '''
    Read the Yelp Review dataset and vectorize them using SpaCy's GloVe model.
    Save the model in .parquet format to a desired location.

    Input:  dataset_read_path
                Location of the .csv file containing reviews.
            dataset_save_path
                Location of the .parquet file containing vectors you want to save.
    Output: None.
    '''

    # Read the dataset 10 000 entries at a time and generae vectors for them
    df_iter = pd.read_csv(dataset_read_path, chunksize=CHUNKSIZE)
    
    spacy_nlp = spacy.load('en_core_web_lg')

    for i, df in enumerate(df_iter):

        # Convert dataframe to a list.
        texts = df['text'].tolist()
        
        # List to store tokenized vectors.
        tokenized_vectors = []

        # Start time.
        start = time.time()

        
        print (f'Chunk {i}:') 

        # Disables parts of the NLP pipeline that are not used.
        disable = ['ner', 'parser', 'attribute_ruler', 'lemmatizer', 'tagger']

        batch_size = 10

        # Tokenize sentences.
        for doc in tqdm(spacy_nlp.pipe(texts, 
                                batch_size=batch_size, 
                                disable=disable, 
                                n_process=1)):
            # Record the tokens that have a vector.
            sentence = []
            for token in doc:
                if token.has_vector:
                    sentence.append(token.vector)

            # Append the vectorized sample.
            tokenized_vectors.append(sentence)

        # End time.
        end = time.time()


        print (f'{end-start:.2f}s')

        # Save the data in .parquet format, an efficient file read/write format.
        df['text_vectorized'] = tokenized_vectors
        df['text_vectorized'] = df['text_vectorized'].apply(lambda v: np.vstack(v))
        df['target'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        
        print (f'\tWriting chunk {i}...')

        np.save(f'./data/features/X_{i}.npy', df['text_vectorized'].values)
        np.save(f'./data/labels/y_{i}.npy', df['target'].values)