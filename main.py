import pandas as pd
import spacy
from tqdm import tqdm
import time

spacy_nlp = spacy.load('en_core_web_lg')

IN_DATA_PATH = './data/yelp_review_100k.csv'
OUT_DATA_PATH = './data/vectorized_yelp_dataset_100k.parquet'

CHUNKSIZE = 10000

def main():
    df_iter = pd.read_csv(IN_DATA_PATH, chunksize=CHUNKSIZE)

    for i, df in enumerate(df_iter):
    
        # Read Dataset
        X = df['text']
        y = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

        # Create word vectors using Spacy
        texts = X.tolist()
        
        # Create a list to store token word vectors with ownership information (sample identification)
        tokenized_vectors = []

        start = time.time()

        batch_size=10
        disable = ['ner', 'parser', 'attribute_ruler', 'lemmatizer', 'tagger']
        print (f'Chunk {i}:')
        for doc in tqdm(spacy_nlp.pipe(texts, 
                                batch_size=batch_size, 
                                disable=disable, 
                                n_process=1)):
            sentence = []
            for token in doc:
                if token.has_vector:
                    sentence.append(token.vector)
            tokenized_vectors.append(sentence)
        end = time.time()


        print (f'{end-start:.2f}s')
        df['text_vectorized'] = tokenized_vectors
        df.to_parquet(f'./data/vectorized_chunk_{i}.parquet', index=False)



if __name__ == "__main__":
    main()
