import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

def extract_genres_from_description(description):
 
    common_genres = [
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'family', 'fantasy', 'horror', 'mystery', 'romance', 
        'sci-fi', 'thriller', 'western'
    ]
    

    description_lower = description.lower()
    

    mentioned_genres = [
        genre for genre in common_genres 
        if genre in description_lower or genre.replace('-', ' ') in description_lower
    ]
    
    return mentioned_genres

def load_movie_data():

    try:
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        print(f"Loaded {len(movies_df)} movies from dataset")
        
        movies_df['overview'] = movies_df['overview'].fillna('')
        movies_df['genres'] = movies_df['genres'].fillna('[]')
        movies_df['keywords'] = movies_df['keywords'].fillna('[]')
        
        return movies_df
    
    except FileNotFoundError:
        print("Error: Movie dataset file not found!")
        raise

def process_movie_features(movies_df):


    def extract_genres(genre_str):
        genres = json.loads(genre_str)
        return [genre['name'].lower() for genre in genres]
    
    movies_df['genre_list'] = movies_df['genres'].apply(extract_genres)
    movies_df['genres_text'] = movies_df['genre_list'].apply(lambda x: ' '.join(x))
    

    def extract_keywords(keyword_str):
        keywords = json.loads(keyword_str)
        return ' '.join([keyword['name'].lower() for keyword in keywords])
    
    movies_df['keywords_text'] = movies_df['keywords'].apply(extract_keywords)
    

    movies_df['combined_features'] = (
        movies_df['overview'] + ' ' + 
        movies_df['genres_text'] + ' ' + 
        movies_df['keywords_text']
    )
    
    return movies_df

def create_similarity_matrices(movies_df):
#making vectors
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)  
    )
    
    #simple text similarity
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
    
    return tfidf, tfidf_matrix

def calculate_genre_bonus(movie_genres, user_genres, base_score):

    if not user_genres:
        return 0
    
    matching_genres = len(set(movie_genres) & set(user_genres))

    # also put in bonus score for multple genres
    bonus = matching_genres * 0.1 * base_score
    return bonus

def get_movie_recommendations(user_description, movies_df, tfidf, tfidf_matrix, n_recommendations=5):

    # extract genres from user description
    user_genres = extract_genres_from_description(user_description)
    print(f"\nDetected genres from your description: {', '.join(user_genres) if user_genres else 'None'}")
    
    # get text similarity scores
    user_vector = tfidf.transform([user_description])
    base_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # calc final scores with genre bonus
    final_scores = np.zeros_like(base_similarities)
    for idx, base_score in enumerate(base_similarities):
        genre_bonus = calculate_genre_bonus(
            movies_df.iloc[idx]['genre_list'],
            user_genres,
            base_score
        )
        final_scores[idx] = base_score + genre_bonus
    
    # get top recommendations
    top_indices = final_scores.argsort()[-n_recommendations:][::-1]
    
    recommendations = []
    for idx in top_indices:
        movie = movies_df.iloc[idx]
        recommendations.append({
            'title': movie['original_title'],
            'genres': ', '.join(movie['genre_list']),
            'score': final_scores[idx],
            'overview': movie['overview'][:100] + '...' if len(movie['overview']) > 100 else movie['overview']
        })
    
    return recommendations

def main():
    print("Initializing Movie Recommendation System...")
    
    movies_df = load_movie_data()
    movies_df = process_movie_features(movies_df)
    tfidf, tfidf_matrix = create_similarity_matrices(movies_df)
    

    while True:
        user_input = input("\nWhat kind of movie are you looking for? (type 'quit' to exit): ")

        if user_input.lower() == 'quit':
            break    
        
        recommendations = get_movie_recommendations(
            user_input, movies_df, tfidf, tfidf_matrix
        )
        
        print("\nTop Recommendations for You:")
        print("-----------------------------")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Match Score: {rec['score']:.3f}")

if __name__ == "__main__":
    main()