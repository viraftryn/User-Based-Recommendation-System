from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_clean = pd.read_pickle('df_clean.pkl')
user_label_encoder = joblib.load('user_label_encoder.pkl')
product_label_encoder = joblib.load('product_label_encoder.pkl')
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class RecommendationRequest(BaseModel):
    user_id: str  
    num_recommendations: int = 3 

@app.post("/recommendations/")
async def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    n = request.num_recommendations

    try:
        encoded_user_id = user_label_encoder.transform([user_id])[0]
        recommendations = get_recommendation_user_purchases(df_clean, encoded_user_id, n)
        return {"recommendations": recommendations.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_popularity_score(df, product_id):
    return df[df['product_id'] == product_id].shape[0]

def get_recommendation_user_purchases(df, user_id, n):
    user_purchases = df[df['user_id'] == user_id]
    user_tags = user_purchases['Tags'].tolist()

    user_profile = ', '.join(user_tags)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Tags'].tolist() + [user_profile])

    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similarity_scores = cosine_sim[0]

    popularity_scores = df['product_id'].apply(lambda x: get_popularity_score(df, x))

    # Combine similarity scores with popularity (e.g., 70% similarity, 30% popularity)
    combined_scores = 0.7 * similarity_scores + 0.3 * popularity_scores

    similar_indices = combined_scores.argsort()[-n:][::-1]

    recommended_products = df.iloc[similar_indices][['product_id', 'product_name']]

    return recommended_products
