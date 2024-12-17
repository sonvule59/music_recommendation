import sys
sys.path.append('Music_Recommendation_System')
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Example user-song interaction matrix
interactions = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4],
    'song_id': [1, 2, 2, 3, 1, 3, 2],
    'rating': [5, 3, 4, 2, 1, 4, 5]
})

def collaborative_filter(ratings_matrix, k=2):
    """
    Perform collaborative filtering using SVD
    
    Parameters:
    -----------
    ratings_matrix : numpy.ndarray
        Matrix of user-item ratings
    k : int
        Number of latent factors to use
        
    Returns:
    --------
    numpy.ndarray
        Matrix of predicted ratings
    """
    if ratings_matrix.size == 0:
        raise ValueError("Empty ratings matrix")
        
    if k < 1:
        raise ValueError("k must be at least 1")
        
    if k > min(ratings_matrix.shape):
        k = min(ratings_matrix.shape) - 1
        print(f"Warning: k was larger than matrix dimension. Using k={k}")
    
    # Normalize the ratings matrix
    ratings_mean = np.mean(ratings_matrix, axis=1)
    ratings_norm = ratings_matrix - ratings_mean.reshape(-1, 1)
    
    # Singular Value Decomposition
    U, sigma, Vt = svds(ratings_norm, k=k)
    
    # Reconstruct the matrix
    sigma_diag = np.diag(sigma)
    predictions = np.dot(np.dot(U, sigma_diag), Vt) + ratings_mean.reshape(-1, 1)
    
    # Ensure predictions are within valid range
    predictions = np.clip(predictions, 0, 5)
    
    return predictions
def get_recommendations(user_id, predictions_df, items_df, n_recommendations=5):
    # Get user's predictions
    user_predictions = predictions_df.loc[user_id]
    
    # Get only songs that user hasn't rated yet
    user_actual_ratings = items_df.loc[user_id]
    unrated_items = user_predictions[user_actual_ratings == 0]
    
    # Sort predictions and get top N
    recommendations = unrated_items.sort_values(ascending=False)
    
    if len(recommendations) < n_recommendations:
        print(f"Warning: Only {len(recommendations)} items available for recommendation")
        
    return recommendations[:n_recommendations]
#     user_predictions = predictions_df.loc[user_id].sort_values(ascending=False)
    
#     # Get only songs that user hasn't rated yet
#     songs_user_hasnt_rated = user_predictions[matrix.loc[user_id] == 0]
    
#     return songs_user_hasnt_rated[:n_recommendations]

# # Convert to user-item matrix
# matrix = interactions.pivot(index='user_id', columns='song_id', values='rating').fillna(0)

def content_based_similarity(song_features):
    return cosine_similarity(song_features)

# Example with song features
song_features = pd.DataFrame({
    'song_id': [1, 2, 3],
    'genre': ['rock', 'pop', 'jazz'],
    'tempo': [120, 100, 90],
    'energy': [0.8, 0.6, 0.4]
})

def hybrid_recommendations(user_id, collaborative_predictions, content_similarities, 
                         alpha=0.7):
    # Combine both predictions with weight alpha
    hybrid_scores = (alpha * collaborative_predictions + 
                    (1 - alpha) * content_similarities)
    return hybrid_scores

# song_features = pd.DataFrame({
#     'song_id': [1, 2, 3],
#     'genre': ['rock', 'pop', 'jazz'],
#     'tempo': [120, 100, 90],
#     'energy': [0.8, 0.6, 0.4]
# })


