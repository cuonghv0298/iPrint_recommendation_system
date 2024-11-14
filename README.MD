# Problem Statement

iPrint, an Indian media house, provides diverse news and information services across sports, weather, health, stocks, and more. While iPrint has historically recommended popular and similar articles to users, this approach lacks personalization and fails to engage many users, resulting in declining user retention and revenue. To address this, iPrint aims to deliver a more personalized experience by recommending new, relevant articles to users each day and suggesting similar articles on individual news pages.

To meet these goals, iPrint seeks a robust recommendation system that will:

1.  Display the top 10 new relevant articles to users at the start of each day.
2. Recommend the top 10 similar articles when users click on any news item.

The recommendation system must avoid showing content previously viewed or removed from the platform, and only English-language articles will be used for content-based recommendations. The final output should provide the names and IDs of the recommended articles.

# Data Sources
The recommendation system will use the following datasets:

1. consumer_transactions: Contains transaction details of consumer interactions with news articles on the platform.

event_timestamp: Timestamp of the user's interaction with the article.

interaction_type: Type of user interaction, such as content_commented_on, content_followed, 
content_liked, content_saved, or content_watched.

item_id: Unique ID of the interacted item.

consumer_id: Unique ID of the consumer who interacted with the item.

consumer_session_id: Unique session ID during which the interaction occurred.

consumer_device_info: Information on the consumer's device.

consumer_location: Location of the consumer.

country: Country of the consumer.

2. platform_content: Contains details of all news articles available on the platform.

event_timestamp: Timestamp of the interaction.

interaction_type: Type of interaction, either content present or pulled out.

item_id: Unique ID of the news item.

producer_id: ID of the producer who added or interacted with the content.

producer_session_id: ID of the session during which the producer interacted with the content.

producer_device_info: Device information of the producer.

producer_location: Location of the producer.

producer_country: Country of the producer.

item_type: Type of content item, such as HTML, VIDEO, or RICH.

item_url: URL of the item.

title: Title of the content.

text_description: Description of the content.

language: Language of the content (only English articles are used for recommendations).

# Procedure Overview

1. Data Pre-processing
- Impute Ratings: Since the dataset lacks user ratings, generate a "ratings" feature based on the interaction type, assigning higher weights to types indicating greater engagement (e.g., highest to content_followed, then content_commented_on, etc.). This processed dataset will serve as the base for collaborative filtering models.
- Filter English Content: Extract only English-language articles from the platform_content data for content-based filtering.

2. Exploratory Data Analysis (EDA)
- Analyze key features like interaction types, consumer and producer locations, item types, and language distributions.
- Identify trends such as popular content types, common languages, and primary regions for article consumption.

3. Recommendation Techniques
- User-based Collaborative Filtering: Build a user-item matrix using the rating values, then generate a user-similarity matrix. Predict ratings for user-item pairs to make recommendations.
- Item-based Collaborative Filtering: Create an item-similarity matrix to recommend the top 10 similar items based on similarity scores.
- Content-based Filtering: Use text processing (e.g., TF-IDF) on article descriptions to recommend similar items based on content relevance.
- ALS (Alternating Least Squares): Create sparse matrices for users and items, train the ALS model, and fine-tune its hyperparameters to improve recommendation quality.

4. Hybrid Recommendation System
- Combine scores from content-based and collaborative filtering models, assigning appropriate weightings to each. Experiment with model hybrids, such as Content + Item-based or ALS + Content-based, to enhance recommendation accuracy.

5. Model Evaluation
- Evaluate using metrics like RMSE, MAE, and precision@k for user-specific recommendations, and global precision@k for overall system performance.
- For the secondary problem statement, explore online evaluation techniques, considering real-time or dynamic feedback mechanisms to enhance recommendation accuracy in production.