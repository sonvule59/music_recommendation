import unittest
import numpy as np
import pandas as pd
from recommender.main import (
    collaborative_filter,
    get_recommendations,
    content_based_similarity,
    hybrid_recommendations
)

class TestMusicRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across all tests"""
        # Create sample test data
        cls.test_interactions = pd.DataFrame({
            'user_id':  [1, 1, 2, 2, 3, 3, 1],
            'song_id':  [1, 2, 2, 3, 1, 4, 5],
            'rating':   [5, 3, 4, 2, 1, 5, 4]
        })
        
        # Create the user-item matrix
        cls.test_matrix = cls.test_interactions.pivot(
            index='user_id',
            columns='song_id',
            values='rating'
        ).fillna(0)

    def test_matrix_shape(self):
        """Test if the matrix has correct dimensions"""
        expected_users = 3  # We have users 1, 2, and 3
        expected_songs = 5  # We have songs 1, 2, 3, 4, and 5
        
        self.assertEqual(
            self.test_matrix.shape, 
            (expected_users, expected_songs),
            f"Matrix shape {self.test_matrix.shape} does not match expected shape ({expected_users}, {expected_songs})"
        )
        
        # Add additional checks to make test more informative
        unique_users = self.test_interactions['user_id'].nunique()
        unique_songs = self.test_interactions['song_id'].nunique()
        
        self.assertEqual(
            unique_users, 
            expected_users, 
            f"Expected {expected_users} users but found {unique_users}"
        )
        self.assertEqual(
            unique_songs, 
            expected_songs, 
            f"Expected {expected_songs} songs but found {unique_songs}"
        )
        
        # Print debug information
        print("\nMatrix shape debug information:")
        print(f"Test matrix shape: {self.test_matrix.shape}")
        print(f"Unique users: {unique_users}")
        print(f"Unique songs: {unique_songs}")
        print("\nTest matrix contents:")
        print(self.test_matrix)

    def test_collaborative_filter(self):
        """Test collaborative filtering predictions"""
        predictions = collaborative_filter(self.test_matrix.values, k=2)
        
        # Test shape of predictions
        self.assertEqual(predictions.shape, self.test_matrix.shape)
        
        # Test if predictions are within valid range (typically 0-5 for ratings)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 5))

    def test_get_recommendations(self):
        """Test recommendation generation for a specific user"""
        # Get predictions
        predictions = collaborative_filter(self.test_matrix.values)
        predictions_df = pd.DataFrame(
            predictions,
            columns=self.test_matrix.columns,
            index=self.test_matrix.index
        )
        
        # Test recommendations for user 1
        recs = get_recommendations(
            user_id=1,
            predictions_df=predictions_df,
            items_df=self.test_matrix,
            n_recommendations=2
        )
        
        # Test number of recommendations
        self.assertEqual(len(recs), 2)
        f"Expected 2 recommendations but got {len(recs)}. Available recommendations: {recs}"
   
        
        # Test that recommended items weren't already rated by user
        user_rated_items = self.test_matrix.loc[1]
        user_rated_items = user_rated_items[user_rated_items > 0].index
        self.assertTrue(all(item not in user_rated_items for item in recs.index))

    def test_content_based_similarity(self):
        """Test content-based similarity calculations"""
        test_features = pd.DataFrame({
            'song_id': [1, 2, 3],
            'tempo': [120, 100, 90],
            'energy': [0.8, 0.6, 0.4]
        }).set_index('song_id')
        
        similarities = content_based_similarity(test_features)
        
        # Test similarity matrix properties
        self.assertEqual(similarities.shape, (3, 3))  # 3x3 similarity matrix
        self.assertTrue(np.allclose(similarities, similarities.T))  # Symmetric
        self.assertTrue(np.all(similarities >= 0))  # Non-negative
        self.assertTrue(np.all(similarities <= 1))  # Bounded by 1

    def test_hybrid_recommendations(self):
        """Test hybrid recommendation system"""
        # Create test predictions and similarities
        collaborative_preds = np.array([[0.8, 0.6], [0.4, 0.7]])
        content_sims = np.array([[0.9, 0.5], [0.5, 0.8]])
        
        hybrid_scores = hybrid_recommendations(
            user_id=1,
            collaborative_predictions=collaborative_preds,
            content_similarities=content_sims,
            alpha=0.7
        )
        
        # Test shape
        self.assertEqual(hybrid_scores.shape, collaborative_preds.shape)
        
        # Test if weights are properly applied
        expected_scores = 0.7 * collaborative_preds + 0.3 * content_sims
        self.assertTrue(np.allclose(hybrid_scores, expected_scores))

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test empty matrix
        empty_matrix = np.array([[]])
        with self.assertRaises(ValueError):
            collaborative_filter(empty_matrix)
        
        # Test invalid k value
        with self.assertRaises(ValueError):
            collaborative_filter(self.test_matrix.values, k=0)
        
        # Test invalid user_id
        predictions_df = pd.DataFrame(
            collaborative_filter(self.test_matrix.values),
            columns=self.test_matrix.columns,
            index=self.test_matrix.index
        )
        with self.assertRaises(KeyError):
            get_recommendations(
                user_id=999,  # Non-existent user
                predictions_df=predictions_df,
                items_df=self.test_matrix
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)