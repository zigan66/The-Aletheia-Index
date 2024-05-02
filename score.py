# score.py can be used to generate labels for training data if you would like to fine-tune the model with a new dataset.


import pandas as pd
import numpy as np
from feature_extraction import extract_features_dataframe

# Define the weights for each feature category globally
weights = {
    # A
    'factuality': 0.1,
    'objectivity': 0.1,
    'comprehensiveness': 0.1,
    'depth': 0.05,
    'language': 0.15,
    'structure': 0.15,
    'headline': 0.15,
    'credibility': 0.1,
    'bias': 0.1
}


def calculate_score(features, weights):
    try:
        # Map each feature to its category and calculate the weighted score
        # Update the features with their scaled names as per the CSV
        weighted_scores = {
            'factuality': (features['scaled_num_entities'] + features['scaled_num_quotes'] - features['scaled_num_speculations']) * weights['factuality'],
            'objectivity': (1 - features['scaled_sentiment_subjectivity']) * weights['objectivity'],
            'comprehensiveness': features['topic_score'] * weights['comprehensiveness'],  # Assuming 'scaled_topic_score' exists
            'depth': (features['scaled_avg_sentence_length'] + features['scaled_article_length']) * weights['depth'],
            'language': features['scaled_num_adjectives'] * weights['language'],
            'structure': features['scaled_readability'] * weights['structure'],
            'headline': features['scaled_headline_relevance'] * weights['headline'],
            'credibility': (features['scaled_num_numerical_data'] + features['scaled_num_speculations']) * weights['credibility'],
            'bias': (1 - features['scaled_sentiment_polarity']) * weights['bias']
        }

        # Sum the weighted scores for the final score
        final_score = sum(weighted_scores.values())

        # Normalize the final score to be out of 1
        normalized_final_score = final_score / sum(weights.values())

        return normalized_final_score
    except Exception as e:
        print(f"Error calculating score: {e}")
        return np.nan




def process_articles_and_score(input_csv, output_csv):
    # Extract features to DataFrame
    articles_df = pd.read_csv(input_csv)
    
    # Rename columns to include 'scaled_' prefix based on what was generated in feature_extraction.py
    scaled_columns = {col: f'scaled_{col}' for col in weights.keys()}
    articles_df.rename(columns=scaled_columns, inplace=True)
    
    # Calculate the trustworthy rate for each article using the renamed, scaled columns
    articles_df['trustworthy_rate'] = articles_df.apply(lambda row: calculate_score(row.to_dict(), weights), axis=1)
    
    # Save the DataFrame to CSV
    articles_df.to_csv(output_csv, index=False)
    print(f"Processed articles with scores saved to {output_csv}")

# Main execution
if __name__ == "__main__":
    input_csv_path = "/Users/jamiezhao/Desktop/articles_with_features.csv"  # Update this path if necessary
    output_csv_path = "/Users/jamiezhao/Desktop/articles_with_scores.csv"
    process_articles_and_score(input_csv_path, output_csv_path)
