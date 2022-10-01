import pandas as pd
import numpy as np
from helpers import unicode_to_str, translate_to_eng, remove_non_alpha, clean_orig_dataset, clean_review
from unsupervised_sa_predict import check_sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")


# Load your dataset.
# Assuming that  NEW/Evaluation data is in same format that of
# London_hotel_reviews - London_hotel_reviews.csv.
# Change the below path to your evaluation dataset path.
df = pd.read_csv('<path-to-evalution-set>')
#df = pd.read_csv('London_hotel_reviews - London_hotel_reviews.csv') - for UT.

#df = df.head(10) #for Unit testing.
df = clean_orig_dataset(df)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


# Load the model.
model = joblib.load('random_forest_model')
print(model.n_features_in_ )

# TF-IDF Vectorization.
tfidf_vect = TfidfVectorizer(analyzer=(lambda x : clean_review(ps, stopwords, x)))
tfidf_vect_fit = tfidf_vect.fit(df['Translated Review'])
tfidf_newdata = tfidf_vect_fit.transform(df['Translated Review'])

# Set the feature dimesion of the input to match that of the model.
tfidf_newdata._shape = (tfidf_newdata.shape[0], model.n_features_in_)

X_tfidf_newdata = pd.DataFrame(tfidf_newdata.toarray())


y_pred = model.predict(X_tfidf_newdata)
print(f'The model predicts the sentiment as : {y_pred}')

