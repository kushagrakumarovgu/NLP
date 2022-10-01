import pandas as pd
import numpy as np
from helpers import unicode_to_str, translate_to_eng, remove_non_alpha, clean_orig_dataset
from unsupervised_sa_predict import check_sentiment

# Load your dataset.
# Assuming that  NEW/Evaluation data is in same format that of
# London_hotel_reviews - London_hotel_reviews.csv.
# Change the below path to your evaluation dataset path.
df = pd.read_csv('<path-to-evalution-set>')
#df = pd.read_csv('London_hotel_reviews - London_hotel_reviews.csv') - for UT.

#df = df.sample(3) #for testing.

df = clean_orig_dataset(df)

#print(len(df))
# For generating reviews for the entire data.
df['Translated Review'].apply(check_sentiment)

# for predicting on single a review uncomment below code and set 
# the index to specific review.
#check_sentiment(df['Translated Review'].iloc[<pass_an_index>].to_string())






