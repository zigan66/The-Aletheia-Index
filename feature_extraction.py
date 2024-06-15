# adjust input and output file location before run the code

import spacy
from scipy.sparse import hstack
import scipy.linalg
import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import gensim.corpora as corpora
from gensim.models import HdpModel
from collections import Counter
import os
import json


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the text and extract features for topic modeling
def preprocess_text_for_lda(text):
    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # Preprocessing the text
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ']]


def extract_topics(text, num_topics=3, num_words=5):
    preprocessed_text = ' '.join(preprocess_text_for_lda(text))
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([preprocessed_text])
    lda = LDA(n_components=num_topics, max_iter=30, learning_method='online', random_state=0)
    lda.fit(doc_term_matrix)
    terms = vectorizer.get_feature_names_out()

    # Collect all topics' keywords
    all_keywords = []
    topics = {}
    for idx, topic in enumerate(lda.components_):
        topic_keywords = [terms[i] for i in topic.argsort()[:-num_words-1:-1]]
        all_keywords.append(topic_keywords)
        topics[f"Topic {idx + 1}"] = topic_keywords

    # Calculate the topic_score based on unique words in topics
    unique_keywords = set(all_keywords[0])
    for keywords in all_keywords[1:]:
        unique_keywords.intersection_update(keywords)

    topic_score = 1 if len(unique_keywords) > 2 else len(unique_keywords) / num_topics

    return topics, topic_score


# Function to generate text summary
def summarize_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    vectorizer = TfidfVectorizer(stop_words='english')
    sent_vectors = vectorizer.fit_transform(sentences)
    summed_vectors = sent_vectors.sum(axis=0).A[0]  # Flatten the summed_vectors

    # Debug: Print the contents and shape
    print("Summed vectors:", summed_vectors)

    if len(sentences) < 3:
        print("Not enough sentences to summarize.")
        return "Not enough data to generate a summary."  # Handling cases with fewer than 3 sentences

    important_indices = summed_vectors.argsort()[-3:][::-1]
    print("Important indices:", important_indices)

    # Ensure we don't exceed the number of available sentences
    important_indices = [idx for idx in important_indices if idx < len(sentences)]
    summary = ' '.join(sentences[idx] for idx in important_indices)
    return summary

# Function to calculate headline relevance
def headline_relevance(headline, summary):
    # Vectorizer for unigrams
    vectorizer_unigram = TfidfVectorizer(stop_words='english')
    unigram_headline = vectorizer_unigram.fit_transform([headline])
    unigram_summary = vectorizer_unigram.transform([summary])

    # Vectorizer for bigrams
    vectorizer_bigram = TfidfVectorizer(stop_words='english', ngram_range=(2,2))
    bigram_headline = vectorizer_bigram.fit_transform([headline])
    bigram_summary = vectorizer_bigram.transform([summary])

    # Stack unigram and bigram vectors
    vectors_headline = hstack([unigram_headline, bigram_headline])
    vectors_summary = hstack([unigram_summary, bigram_summary])

    # Calculate cosine similarity
    similarity = cosine_similarity(vectors_headline, vectors_summary)[0, 0]
    return similarity


# Feature extraction function that handles data row-wise
def extract_features(row):
    try:
        text = row['article_text']  
        title = row.get('title', '')  
        doc = nlp(text)
        summary = summarize_text(text)
        relevance = headline_relevance(title, summary) if title else 0
        topics, topic_score = extract_topics(text)

        features = {
            'num_entities': len(doc.ents),
            'num_quotes': sum(token.tag_ == "''" or token.tag_ == '``' for token in doc),
            'sentiment_polarity': TextBlob(text).sentiment.polarity,
            'sentiment_subjectivity': TextBlob(text).sentiment.subjectivity,
            'num_speculations': sum(1 for token in doc if token.lemma_ in ['might', 'could', 'possibly']),
            'article_length': len(text),
            'avg_sentence_length': sum(len(sent.text) for sent in doc.sents) / len(list(doc.sents)),
            'num_adjectives': sum(token.pos_ == 'ADJ' for token in doc),
            'num_numerical_data': sum(token.like_num for token in doc),
            'readability': textstat.flesch_reading_ease(text),
            'headline_relevance': relevance,
            'topic_score': topic_score # Add topic score to the features
        }
        # print("Banana features", features)
        return pd.Series(features)
    except Exception as e:
        print(f"An error occurred while processing text: {e}")
        return pd.Series()
    

def normalize_features(df, feature_names):
    for feature in feature_names:
        max_value = df[feature].max()
        min_value = df[feature].min()
        df[f'scaled_{feature}'] = (df[feature] - min_value) / (max_value - min_value)

    return df

def serialize_features_to_json(df, output_json_path):
    # List of keys to include in the JSON output
    keys_to_include = [
        'id', 'title', 'platform', 'article_text',
        'scaled_num_entities', 'scaled_num_quotes', 'scaled_sentiment_polarity', 
        'scaled_sentiment_subjectivity', 'scaled_num_speculations', 
        'scaled_article_length', 'scaled_avg_sentence_length', 
        'scaled_num_adjectives', 'scaled_num_numerical_data', 
        'scaled_readability', 'scaled_headline_relevance','topic_score'
    ]

    # Prepare data for JSON
    json_data = {
        "instances": [
            {key: instance[key] for key in keys_to_include if key in instance}
            for instance in df.to_dict(orient='records')
        ]
    }

    # Ensure the directory exists where the JSON will be saved
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Write JSON data to file
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON output saved to {output_json_path}")
    
def extract_features_dataframe(input_csv):
    df = pd.read_csv(input_csv)
    features_df = df.apply(extract_features, axis=1)
    enhanced_df = pd.concat([df, features_df], axis=1)
    return enhanced_df


def process_file(csv_path, output_json_path='output.json'):
    try:
        df = pd.read_csv(csv_path)
        features_df = df.apply(extract_features, axis=1)
        enhanced_df = pd.concat([df, features_df], axis=1)
        normalized_df = normalize_features(enhanced_df, [
            'num_entities', 'num_quotes', 'sentiment_polarity', 'sentiment_subjectivity', 
            'num_speculations', 'article_length', 'avg_sentence_length', 'num_adjectives', 
            'num_numerical_data', 'readability', 'headline_relevance'
        ])
        serialize_features_to_json(normalized_df, output_json_path)
        return output_json_path
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None


# Main execution block
if __name__ == "__main__":
    input_csv_path = "./articles.csv"
    output_json_path = "./articles.json"
    
    # Extract and process features
    df = extract_features_dataframe(input_csv_path)
    print("Features extracted successfully.")
    
    # Normalize the features
    feature_names = [
        'num_entities', 'num_quotes', 'sentiment_polarity', 'sentiment_subjectivity', 
        'num_speculations', 'article_length', 'avg_sentence_length', 'num_adjectives', 
        'num_numerical_data', 'readability', 'headline_relevance'
    ]
    df_normalized = normalize_features(df, feature_names)

    # Output to JSON
    serialize_features_to_json(df_normalized, output_json_path)






