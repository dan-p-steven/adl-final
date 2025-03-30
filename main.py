import pandas as pd
import numpy as np
import src.preprocess as preprocess

DATABASE_PATTERN = './data/*.parquet'


def main():
    preprocess.generate_save_word_embeddings(dataset_read_path='./data/yelp_review_100k.csv')


if __name__ == "__main__":
    main()
