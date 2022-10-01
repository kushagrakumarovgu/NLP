import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from helpers import label_reviews, remove_non_alpha
import warnings

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon")


def sa_vader(review: str):

    """Returns Sentiments of text/review (pos, neg or neu) using nltk VADER"""

    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(review)["compound"]

    if score >= 0.05:
        return "pos"
    elif score <= -0.05:
        return "neg"
    else:
        return "neu"


def sa_textblob(reviews: str):

    """
    Returns Sentiments of text/review (pos, neg or neu) using TEXTBLOB
    """

    score = TextBlob(reviews).sentiment.polarity

    if score > 0:
        return "pos"
    elif score == 0:
        return "neu"
    else:
        return "neg"


def sa_bert(model, tokenizer, review: str):

    """
    Returns Sentiments of text/review (pos, neg or neu) using transformer BERT Model.
    """

    tokens = tokenizer.encode(
        review,
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
        return_tensors="pt",
    )

    result = model(tokens)
    score = int(torch.argmax(result.logits)) + 1

    if score <= 2:
        return "neg"
    elif score == 3:
        return "neu"
    else:
        return "pos"


def check_sentiment(review: str):

    """
    Computes sentiment of a text / reviews using 3 models (VADER, TEXTBLOB, BERT)
    and returns sentiment (pos, neg or neu) based on majority voting.
    """

    # Extract alphabets from the review.
    review = re.sub("[^a-zA-Z]", " ", review).strip()

    if review:

        vader_res = sa_vader(review)

        txtblob_res = sa_textblob(review)

        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )

        bert_res = sa_bert(model, tokenizer, review)

        result_list = [bert_res, vader_res, txtblob_res]

        # choose the sentiment predicted/voted most.
        # NOTE: In case of Tie, the priority is given
        # to the Bert model (1st entry in the list).
        max_vote = max(result_list, key=result_list.count)

        if max_vote == "pos":
            print("The sentiment of the review is Postive")
        elif max_vote == "neg":
            print("The sentiment of the review is Negative")
        else:
            print("The sentiment of the review is Neutral")

    else:
        print("Error: The review has NO english alphabets. Please check again")


def individual_accuaracy(model_name, sentiment):

    return round(
        len(df[(df[model_name] == sentiment) & (df["label"] == sentiment)])
        / len(df[df["label"] == sentiment])
        * 100,
        2,
    )


if __name__ == "__main__":
    df = pd.read_csv("Translated_London_Hotel_review_Dataset.csv")
    print(df.columns)

    print(df.head())

    df["Translated Review"] = df["Translated Review"].apply(remove_non_alpha)

    # Set empty cells to Nan, so that we can drop.
    df["Translated Review"].replace("", np.nan, inplace=True)

    print(f"{df.isna().sum()}")

    df = df.dropna()

    print(f"{df.isna().sum()}")

    df["label"] = df["Review Rating"].apply(label_reviews)

    print(df.head())

    df["vader_prediction"] = df["Translated Review"].apply(sa_vader)

    df["textblob_prediction"] = df["Translated Review"].apply(sa_textblob)

    # NOTE: This might take close to an hour or more
    # depending on the system capacity.
    df["Bert_prediction"] = df["Translated Review"].apply(sa_bert)

    print(df.head())

    ## VADER EVALUATION

    print(
        f"VADER Model Overall accuracy: {accuracy_score(df['label'], df['vader_prediction']) * 100}"
    )

    print(
        f'VADER accuracy on neutral sentiments: {individual_accuaracy("vader_prediction", "neu")}'
    )

    print(
        f'VADER accuracy on positive sentiments: {individual_accuaracy("vader_prediction", "pos")}'
    )

    print(
        f'VADER accuracy on Negative sentiments: {individual_accuaracy("vader_prediction", "neg")}'
    )

    ## TEXTBLOB EVALUATION

    print(
        f"TEXTBLOB Model Overall accuracy: {accuracy_score(df['label'], df['textblob_prediction']) * 100}"
    )

    print(
        f'TEXTBLOB accuracy on neutral sentiments: {individual_accuaracy("textblob_prediction", "neu")}'
    )

    print(
        f'TEXTBLOB accuracy on positive sentiments: {individual_accuaracy("textblob_prediction", "pos")}'
    )

    print(
        f'TEXTBLOB accuracy on Negative sentiments: {individual_accuaracy("textblob_prediction", "neg")}'
    )

    ## BERT EVALUATION

    print(
        f"BERT Model Overall accuracy: {accuracy_score(df['label'], df['Bert_prediction']) * 100}"
    )

    print(
        f'Bert accuracy on neutral sentiments: {individual_accuaracy("Bert_prediction", "neu")}'
    )

    print(
        f'Bert accuracy on positive sentiments: {individual_accuaracy("Bert_prediction", "pos")}'
    )

    print(
        f'Bert accuracy on Negative sentiments: {individual_accuaracy("Bert_prediction", "neg")}'
    )
