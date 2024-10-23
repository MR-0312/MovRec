import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.dat', delimiter='::', engine='python', names=['movieId', 'title', 'genres'])
    ratings = pd.read_csv('ratings.dat', delimiter='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    ratings.drop('timestamp', axis=1, inplace=True)  # Remove timestamp if not needed
    return movies, ratings

@st.cache_data
def preprocess_data(movies):
    movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' ').lower())
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

@st.cache_resource
def train_collaborative_filtering(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

def get_content_based_recommendations(movie_title, cosine_sim, movies):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['movieId', 'title']].iloc[movie_indices]

def hybrid_recommendation(user_id, movie_title, model, cosine_sim, movies, ratings):
    cf_predictions = []
    for movieId in movies['movieId'].unique():
        pred = model.predict(user_id, movieId).est
        cf_predictions.append((movieId, pred))
    
    cf_predictions = sorted(cf_predictions, key=lambda x: x[1], reverse=True)[:10]
    
    content_recommendations = get_content_based_recommendations(movie_title, cosine_sim, movies)
    
    movie_ids_cf = [movie[0] for movie in cf_predictions]
    movie_ids_content = content_recommendations['movieId'].values
    hybrid_movie_ids = list(set(movie_ids_cf).union(set(movie_ids_content)))
    
    return movies[movies['movieId'].isin(hybrid_movie_ids)].head(10)

movies, ratings = load_data()
cosine_sim = preprocess_data(movies)
model = train_collaborative_filtering(ratings)

st.title('Movie Recommendation System')
st.write("This app provides movie recommendations using both collaborative filtering and content-based filtering.")

user_id = st.number_input('Enter User ID:', min_value=1, max_value=ratings['userId'].max(), value=1)
movie_title = st.selectbox('Select a Movie You Like:', movies['title'].unique())

if st.button('Recommend Movies'):
    recommendations = hybrid_recommendation(user_id, movie_title, model, cosine_sim, movies, ratings)
    
    st.write(f"Top recommended movies for User {user_id} based on '{movie_title}':")
    for idx, row in recommendations.iterrows():
        st.write(f"- {row['title']}")
