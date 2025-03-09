import pandas as pd
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

def make_features(config):
    print("Making features...")
    train_df = pd.read_csv(config.data.train_csv_save_path)
    test_df = pd.read_csv(config.data.test_csv_save_path)

    vectorizer_name = config.feature.vectorizer
    vectorizer = {
        "count-vectorizer": CountVectorizer,
        "tfidf-vectorizer": TfidfVectorizer
    }[vectorizer_name](stop_words="english")

    train_inputs = vectorizer.fit_transform(train_df["review"])
    # transform because it is a test set, not training the dataset anymore
    # used the prepared vectorizer to transform test data
    test_inputs = vectorizer.transform(test_df["review"])

    # save the result
    joblib.dump(train_inputs, config.feature.train_features_save_path)
    joblib.dump(test_inputs, config.feature.test_features_save_path)


if __name__ = "__main__":
    config = OmegaConf.load("./params.yaml")
    make_features(config)