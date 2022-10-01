import re
import string
import numpy as np
from googletrans import Translator


def label_reviews(rating : str):

    """Label the Review ratings in to positive, negative and neutral"""

    if rating <= 2:
        return "neg"
    elif rating == 3:
        return "neu"
    else:
        return "pos"


def remove_non_alpha(review : str):
    """Remove Non alpha characters for the review"""

    return re.sub("[^a-zA-Z]", " ", review).strip()


def decode_unicode(match):
    code = match.group(1)  # The code in hex
    return chr(int(code, 16))


def unicode_to_str(review : str):
    """Convert unicode input to human readble string format"""
    pattern = "[\w\s]*[a-zA-Z]*[0-9]*-?<U\+([0-9a-fA-F]+)>"
    if re.match(pattern, review):
        review = re.sub(pattern, decode_unicode, review)

    return review


def translate_to_eng(review : str):
    """Detect and convert reviews to english language"""
    trans = Translator()
    if trans.detect(review).lang != "en":
        review = trans.translate(review).text
    return review

def compute_punct_percent(review : str):
    """ Reeturn % of punctuation in a review"""
    count = sum([1 for char in review if char in string.punctuation])
    return round(count/(len(review) - review.count(" ")), 3)*100

def clean_review(ps, stopwords : list, review : str):

    review = remove_non_alpha(review)
    review = "".join([word.lower() for word in review if word not in string.punctuation])
    tokens = re.split('\W+', review)
    review = [ps.stem(word) for word in tokens if word not in stopwords]
    return review


def clean_orig_dataset(df):

    df['Complete Review'] = df['Review Title'] + " " + df['Review Text']

    df['Complete Review'] = df['Complete Review'].apply(unicode_to_str)

    df['Translated Review'] = df['Complete Review'].apply(translate_to_eng)

    df['Translated Review'] = df['Translated Review'].apply(remove_non_alpha)

    # Set empty cells to Nan, so that we can drop.
    df['Translated Review'].replace('', np.nan, inplace=True)
    df = df.dropna()

    return df


