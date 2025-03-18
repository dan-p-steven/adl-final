import pandas as pd

output_file_location = './data/yelp_review_dataset.csv'
input_file_location = './data/yelp_academic_dataset_review.json'
chunk_size = 10000

def main():

    # Read in chunks and write to CSV

    for i, chunk in enumerate(pd.read_json(input_file_location, lines=True, 
                                           chunksize=chunk_size)):

        chunk = chunk[['text', 'stars']]
        # Append to file after the first write
        chunk.to_csv(output_file_location, index=False, mode="a", header=(i == 0))

        print(f"Processed chunk {i+1}")

        if (i > 50):
            break




if __name__ == "__main__":
    main()
