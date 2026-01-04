from flask import Flask, render_template, request
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

books_clean = pickle.load(open("books_clean.pkl", "rb"))
book_embeddings = pickle.load(open("book_embeddings.pkl", "rb"))
title_embeddings = pickle.load(open("title_embeddings.pkl", "rb"))
similarity_matrix = pickle.load(open("similarity_matrix.pkl", "rb"))

model = SentenceTransformer("embedding_model")

# =============================================================================
# RECOMMENDATION FUNCTIONS
# =============================================================================

def recommend_by_embedding_filtered(book_title, top_n=5, min_ratings=500):
    idx = books_clean.index[books_clean['title'] == book_title][0]

    scores = sorted(
        list(enumerate(similarity_matrix[idx])),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for i, score in scores:
        if i == idx:
            continue
        if books_clean.iloc[i]['ratings_count'] >= min_ratings:
            results.append({
                "title": books_clean.iloc[i]['title'],
                "authors": books_clean.iloc[i]['authors'],
                "image_url": books_clean.iloc[i]['final_image_url'],
                "similarity": round(score, 3)
            })
        if len(results) == top_n:
            break

    return results


def recommend_from_user_text(user_text, top_n=5):
    if user_text is None or len(user_text.strip()) == 0:
        return []

    user_embedding = model.encode(
        user_text,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        [user_embedding],
        book_embeddings
    )[0]

    top_indices = scores.argsort()[::-1][:top_n]

    return [
        {
            "title": books_clean.iloc[idx]['title'],
            "authors": books_clean.iloc[idx]['authors'],
            "image_url": books_clean.iloc[idx]['final_image_url'],
            "similarity": round(scores[idx], 3)
        }
        for idx in top_indices
    ]


def hybrid_embedding_recommender(
    user_input,
    top_n=5,
    min_ratings=500,
    title_threshold=0.80
):
    if user_input is None or len(user_input.strip()) == 0:
        return []

    user_embedding = model.encode(
        user_input,
        normalize_embeddings=True
    )

    title_scores = cosine_similarity(
        [user_embedding],
        title_embeddings
    )[0]

    best_idx = np.argmax(title_scores)
    best_score = title_scores[best_idx]

    # Book detected
    if best_score >= title_threshold:
        detected_title = books_clean.iloc[best_idx]['title']
        return recommend_by_embedding_filtered(
            detected_title, top_n, min_ratings
        )

    # Semantic search
    return recommend_from_user_text(user_input, top_n)

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        recommendations = hybrid_embedding_recommender(query)

    return render_template(
        "index.html",
        recommendations=recommendations,
        query=query
    )

if __name__ == "__main__":
    app.run(debug=True)
