import re
import json
import string
import contractions


import numpy as np
from scipy import sparse

from nltk import ngrams
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

import nltk
nltk.download('stopwords')

def remove_path(df):
    cleaned_text = list()
    for text in df.attack_description.values:
        pattern = r'([A-Za-z]:\\[^\s]*)'
        text = re.sub(pattern, '', text)
        pattern = r'(\/.*?\.[\w:]+)'
        text = re.sub(pattern, '', text)
        text = " ".join([chunk for chunk in text.split()if len(chunk) > 0])
        cleaned_text.append(text)

    df["attack_description"] = cleaned_text
    return df

def expand_contraction_word(df):
    expanded_words = list()
    for text in df.attack_description.values:
        temp_expanded_words = list()
        for word in text.split():
            # using contractions.fix to expand the shortened words
            temp_expanded_words.append(contractions.fix(word))

        temp_expanded_text = ' '.join(temp_expanded_words)
        expanded_words.append(temp_expanded_text)

    df["attack_description"] = expanded_words
    return df


def remove_words_within_bracket(df):
    cleaned_text_list = list()

    for text in df.attack_description.values:
        text = re.sub(r'\([^()]*\)', '', text)
        cleaned_text = " ".join([i.strip() for i in text.split(" ") if len(i) > 0])
        cleaned_text_list.append(cleaned_text)

    df["attack_description"] = cleaned_text_list
    return df

def text_features(df):
    longest_word_length = []
    mean_word_length = []
    length_in_chars = []

    for text in df.attack_description.values:
        longest_word_length.append(len(max(text.split(), key=len)))
        mean_word_length.append(np.mean([len(word) for word in text.split()]))
        length_in_chars.append(len(text))

    longest_word_length = np.array(longest_word_length).reshape(-1, 1)
    mean_word_length = np.array(mean_word_length).reshape(-1, 1)
    length_in_chars = np.array(length_in_chars).reshape(-1, 1)

    return np.concatenate([longest_word_length, mean_word_length, length_in_chars], axis=1)


def count_punctuations(df):
    puncts = []
    punctuations = set(string.punctuation)
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    for text in df.attack_description.values:
        puncts.append(count(text,punctuations))
    return np.array(puncts).reshape(-1,1)


def get_common_ngram_count(df):
    def get_common_ngram_count_from_text(sentence):
        common_unigram_count = 0
        common_bigram_count = 0
        common_trigram_count = 0
        common_fourgram_count = 0
        common_fivegram_count = 0

        unigrams = set(unigram.keys())
        bigrams = set(bigram.keys())
        trigrams = set(trigram.keys())
        fourgrams = set(fourgram.keys())
        fivegrams = set(fivegram.keys())

        for n in (range(1, 6)):
            grams = ngrams(sentence.split(), n)
            for gram in grams:
                temp_phrases = " ".join(gram).strip()

                if temp_phrases in unigrams:
                    common_unigram_count += 1
                elif temp_phrases in bigrams:
                    common_bigram_count += 1
                elif temp_phrases in trigrams:
                    common_trigram_count += 1
                elif temp_phrases in fourgrams:
                    common_fourgram_count += 1
                elif temp_phrases in fivegrams:
                    common_fivegram_count += 1
                else:
                    pass

        return common_unigram_count, common_bigram_count, common_trigram_count, common_fourgram_count, common_fivegram_count

    # Opening JSON file
    with open('data/common_unigram.json', 'r') as openfile:
        unigram = json.load(openfile)

    with open('data/common_bigram.json', 'r') as openfile:
        bigram = json.load(openfile)

    with open('data/common_trigram.json', 'r') as openfile:
        trigram = json.load(openfile)

    with open('data/common_fourgram.json', 'r') as openfile:
        fourgram = json.load(openfile)

    with open('data/common_fivegram.json', 'r') as openfile:
        fivegram = json.load(openfile)

    n_grams_counts = list()
    for text in df.attack_description.values:
        n_grams_counts.append(np.array(get_common_ngram_count_from_text(text)))

    return np.array(n_grams_counts)

def calc_sentiment_scores(df):
    sid = SentimentIntensityAnalyzer()
    neg = []
    neu = []
    pos = []
    compound = []

    for text in df.attack_description.values:
        sentiments = sid.polarity_scores(text)
        neg.append(sentiments['neg'])
        neu.append(sentiments['neu'])
        pos.append(sentiments['pos'])
        compound.append(sentiments['compound'])

    neg = np.array(neg).reshape(-1,1)
    neu = np.array(neu).reshape(-1,1)
    pos = np.array(pos).reshape(-1,1)
    compound = np.array(compound).reshape(-1,1)
    return np.concatenate([neg,  pos, compound], axis = 1)


def stopwords_counter(df):
    stop_words = set(stopwords.words('english'))
    text_stopwords_count = list()

    for text in df.attack_description.values:
        stopwords_count = 0
        word_tokens = word_tokenize(text)
        for word in word_tokens:
            if word in stop_words:
                stopwords_count += 1

        text_stopwords_count.append(stopwords_count)

    return np.array(text_stopwords_count).reshape(-1,1)


def get_glove_vectors(df, embeddings_dict):
    vectors = list()
    for text in df.attack_description.values:
        text_vector = list()
        for word in word_tokenize(text):
            try:
                text_vector.append(embeddings_dict[word.lower()])
            except:
                text_vector.append(np.zeros(50))
        vectors.append(np.average(text_vector, axis=0))

    return np.array(vectors)


def tfidf_w2v(df, idf_dict, embeddings_dict):
    vectors = []
    for text in df.attack_description.values:
        text_vector = list()
        for word in word_tokenize(text):
            try:
                text_vector.append(embeddings_dict[word.lower()])
            except:
                text_vector.append(np.zeros(50))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(text)]
        vectors.append(np.average(text_vector, axis=0, weights=weights))

    return np.array(vectors)


def featurize(train_df, test_df, embedding_type):
    print("Remove path from text....")
    train_df = remove_path(train_df)
    test_df = remove_path(test_df)

    print('Remove words within bracket....')
    train_df = remove_words_within_bracket(train_df)
    test_df = remove_words_within_bracket(test_df)

    print('Contraction....')
    train_df = expand_contraction_word(train_df)
    test_df = expand_contraction_word(test_df)

    print('Text Features....')
    train_text_features = text_features(train_df)
    test_text_features = text_features(test_df)

    print('Punctuation....')
    train_num_punctuations = count_punctuations(train_df)
    test_num_punctuations = count_punctuations(test_df)

    print("NGram Count....")
    train_ngram_count = get_common_ngram_count(train_df)
    test_ngram_count = get_common_ngram_count(test_df)

    print('Sentiment Scores....')
    train_sentiment = calc_sentiment_scores(train_df)
    test_sentiment = calc_sentiment_scores(test_df)

    print("Stopwords Counter....")
    train_stopwords = stopwords_counter(train_df)
    test_stopwords = stopwords_counter(test_df)

    if embedding_type == 'tfidf':
        print('TFIDF Title....')

        tfidf_word = TfidfVectorizer()

        print('TFIDF Word....')
        train_word_features = tfidf_word.fit_transform(train_df.attack_description.values)
        test_word_features = tfidf_word.transform(test_df.attack_description.values)

        normalizer_tfidf = MinMaxScaler()
        train_embedding_features = sparse.csr_matrix(normalizer_tfidf.fit_transform(train_word_features.toarray()))
        test_embedding_features = sparse.csr_matrix(normalizer_tfidf.fit_transform(test_word_features.toarray()))

    if embedding_type == 'glove':
        print('Glove.....')
        embeddings_dict = {}
        with open("vector/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        train_glove = get_glove_vectors(train_df, embeddings_dict)
        test_glove = get_glove_vectors(test_df, embeddings_dict)

        normalizer_glove = MinMaxScaler()
        train_glove = normalizer_glove.fit_transform(train_glove)
        test_glove = normalizer_glove.transform(test_glove)

        train_embedding_features = sparse.csr_matrix(train_glove)
        test_embedding_features = sparse.csr_matrix(test_glove)

    if embedding_type == 'tfidf_glove':
        print('TFIDF and Glove.....')
        embeddings_dict = {}
        with open("vector/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        tfidf = TfidfVectorizer()
        tfidf.fit(train_df.attack_description.values)
        idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

        train_glove = tfidf_w2v(train_df, idf_dict, embeddings_dict)
        test_glove = tfidf_w2v(test_df, idf_dict, embeddings_dict)

        normalizer_glove = MinMaxScaler()
        train_glove = normalizer_glove.fit_transform(train_glove)
        test_glove = normalizer_glove.transform(test_glove)

        train_embedding_features = sparse.csr_matrix(train_glove)
        test_embedding_features = sparse.csr_matrix(test_glove)

    """Train Feature Creation"""
    train_features = np.hstack((train_text_features,
                                train_num_punctuations,
                                train_ngram_count,
                                train_sentiment,
                                train_stopwords,
                                ))
    normalizer = MinMaxScaler()
    train_features = normalizer.fit_transform(train_features)
    train_features = sparse.hstack((
                            train_features,
                            train_embedding_features
                            ))

    """Test Feature Creation"""
    test_features = np.hstack((test_text_features,
                                test_num_punctuations,
                                test_ngram_count,
                                test_sentiment,
                                test_stopwords,
                                ))
    normalizer = MinMaxScaler()
    test_features = normalizer.fit_transform(test_features)
    test_features = sparse.hstack((
                            test_features,
                            test_embedding_features
                            ))
    feature_names = [
        "longest_word_length",
        "mean_word_length",
        "length_in_chars",
        "punctuations_num",
        "common_unigram_count",
        "common_bigram_count",
        "common_trigram_count",
        "common_fourgram_count",
        "common_fivegram_count",
        "sentiment_neg",
        "senitment_pos",
        "sentiment_compound",
        "stopwords_num"
    ]

    return train_features, test_features, feature_names


if __name__ == "__main__":
    import pandas as pd
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    embedding_type = 'tfidf_glove'

    featurize(train_df, test_df, embedding_type)