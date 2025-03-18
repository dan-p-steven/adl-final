import pandas as pd

output_file_location = './data/yelp_review_dataset.csv'
input_file_location = './data/yelp_academic_dataset_review.json'
chunk_size = 10000

def get_sentiment(stars):
    if stars <= 2 :
        return 'negative'
    elif stars == 3:
        return 'neutral'
    else:
        return 'positive'

def main():


    df = pd.read_csv('data/yelp_review_dataset.csv')




if __name__ == "__main__":
    main()
