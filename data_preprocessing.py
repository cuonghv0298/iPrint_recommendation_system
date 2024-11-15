import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
consumer_transactions = pd.read_csv('consumer_transactions.csv')
platform_content = pd.read_csv('platform_content.csv')

# Step 1: Impute Ratings Based on Interaction Type
# Assign a rating score to each interaction type
interaction_weights = {
    'content_followed': 5,
    'content_commented_on': 4,
    'content_liked': 3,
    'content_saved': 2,
    'content_watched': 1
}

# Map interaction weights to consumer transactions
consumer_transactions['rating'] = consumer_transactions['interaction_type'].map(interaction_weights)

# Drop interactions without a defined weight
consumer_transactions = consumer_transactions.dropna(subset=['rating'])

# Step 2: Filter English Content
# Filter platform content to retain only English articles
platform_content = platform_content[platform_content['language'] == 'English']

# Step 3: Merge Datasets
# Merge consumer transactions and platform content on `item_id` to connect interactions with content
merged_data = pd.merge(
    consumer_transactions,
    platform_content[['item_id', 'title', 'text_description', 'item_type', 'language']],
    on='item_id',
    how='inner'
)

# Step 4: Data Cleanup
# Remove articles that have been pulled out by filtering for only active content in platform content
active_content_ids = platform_content[platform_content['interaction_type'] == 'content_present']['item_id']
merged_data = merged_data[merged_data['item_id'].isin(active_content_ids)]

# Step 5: Create User-Item Matrix for Collaborative Filtering
# Pivot the data to create a user-item matrix
user_item_matrix = merged_data.pivot_table(index='consumer_id', columns='item_id', values='rating', fill_value=0)

# Step 6: Prepare Content-Based Features Using TF-IDF on Article Descriptions
# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(platform_content['text_description'].fillna(""))

# Convert TF-IDF matrix to a DataFrame with item IDs as index
content_features = pd.DataFrame(tfidf_matrix.toarray(), index=platform_content['item_id'], columns=tfidf_vectorizer.get_feature_names_out())

# Step 7: Drop Duplicates and Reset Index
# Remove duplicate records if any
merged_data.drop_duplicates(inplace=True)
merged_data.reset_index(drop=True, inplace=True)

# Save preprocessed data for further modeling
user_item_matrix.to_csv('user_item_matrix.csv', index=True)
content_features.to_csv('content_features.csv', index=True)
merged_data.to_csv('merged_data.csv', index=False)

print("Data preprocessing completed. Files saved for collaborative and content-based recommendation models.")
