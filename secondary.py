import pandas as pd
import spacy
from tqdm import tqdm
import time

spacy_nlp = spacy.load('en_core_web_lg')

IN_DATA_PATH = './data/yelp_review_dataset.csv'
OUT_DATA_PATH = './data/yelp_review_100k.csv'

def main():
    df = pd.read_csv(IN_DATA_PATH)
    df = df.iloc[:100000]

    print (df['sentiment'].value_counts())
    df.to_csv(OUT_DATA_PATH, index=False)

if __name__ == "__main__":
    main()
