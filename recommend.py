import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======== =====================================================================
# LOAD DATA
# =============================================================================

books = pd.read_csv("Books.csv")

# Drop duplicates SAFELY
books_clean = books.drop_duplicates(subset='book_id').copy()

# Create semantic text
books_clean['text'] = (
    books_clean['title'].fillna('') + " " +
    books_clean['original_title'].fillna('') + " " +
    books_clean['authors'].fillna('') + " " +
    books_clean['language_code'].fillna('')
)

# Keep only required columns
books_clean = books_clean[
    [
        'book_id', 'title', 'authors',
        'average_rating', 'ratings_count',
        'image_url', 'text'
    ]
].reset_index(drop=True)

# =============================================================================
# IMAGE RESOLVER (ROBUST)
# =============================================================================

def resolve_image(row):
    if pd.notna(row['image_url']) and str(row['image_url']).startswith("http"):
        return row['image_url']

    title = row['title'].replace(" ", "+")
    return f"https://covers.openlibrary.org/b/title/{title}-L.jpg"

books_clean['final_image_url'] = books_clean.apply(resolve_image, axis=1)

# =============================================================================
# LOAD / CREATE EMBEDDING MODEL
# =============================================================================

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Book content embeddings (for recommendation)
book_embeddings = model.encode(
    books_clean['text'].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)
book_embeddings = np.array(book_embeddings)

# Title embeddings (for intent detection)
title_embeddings = model.encode(
    books_clean['title'].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)
title_embeddings = np.array(title_embeddings)

# =============================================================================
# BOOK-TO-BOOK SIMILARITY MATRIX
# =============================================================================

similarity_matrix = cosine_similarity(book_embeddings)

# =============================================================================
# RECOMMENDATION FUNCTIONS
# =============================================================================

def recommend_by_embedding(book_title, top_n=5):
    if book_title not in books_clean['title'].values:
        return []

    idx = books_clean.index[books_clean['title'] == book_title][0]
    scores = sorted(
        list(enumerate(similarity_matrix[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

    return [
        {
            "title": books_clean.iloc[i]['title'],
            "authors": books_clean.iloc[i]['authors'],
            "avg_rating": books_clean.iloc[i]['average_rating'],
            "image_url": books_clean.iloc[i]['final_image_url'],
            "similarity_score": round(score, 3)
        }
        for i, score in scores
    ]


def recommend_by_embedding_filtered(book_title, top_n=5, min_ratings=500):
    if book_title not in books_clean['title'].values:
        return []

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
                "similarity_score": round(score, 3)
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
            "similarity_score": round(scores[idx], 3)
        }
        for idx in top_indices
    ]


def hybrid_embedding_recommender(
    user_input,
    top_n=5,
    min_ratings=500,
    use_popularity_filter=True,
    title_threshold=0.80
):
    """
    EMBEDDING-BASED HYBRID RECOMMENDER

    1. Detect if input is a book title (via title embeddings)
    2. If yes → book-to-book recommendation
    3. Else → semantic search
    """

    if user_input is None or len(user_input.strip()) == 0:
        return []

    # Embed user input once
    user_embedding = model.encode(
        user_input,
        normalize_embeddings=True
    )

    # Compare against title embeddings
    title_scores = cosine_similarity(
        [user_embedding],
        title_embeddings
    )[0]

    best_idx = np.argmax(title_scores)
    best_score = title_scores[best_idx]

    # Case 1: Book detected
    if best_score >= title_threshold:
        detected_title = books_clean.iloc[best_idx]['title']

        if use_popularity_filter:
            return recommend_by_embedding_filtered(
                detected_title,
                top_n=top_n,
                min_ratings=min_ratings
            )
        else:
            return recommend_by_embedding(
                detected_title,
                top_n=top_n
            )

    # Case 2: Semantic search
    return recommend_from_user_text(user_input, top_n)

# =============================================================================
# TEST
# =============================================================================

print(hybrid_embedding_recommender("Twilight"))


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

pickle.dump(books_clean, open("books_clean.pkl", "wb"))
pickle.dump(book_embeddings, open("book_embeddings.pkl", "wb"))
pickle.dump(title_embeddings, open("title_embeddings.pkl", "wb"))
pickle.dump(similarity_matrix, open("similarity_matrix.pkl", "wb"))

model.save("embedding_model")

print("All artifacts saved successfully!")
