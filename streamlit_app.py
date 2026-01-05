import streamlit as st
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download, snapshot_download

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide"
)

st.markdown("""
<style>

/* Page background */
[data-testid="stAppViewContainer"] {
    background-color: #f4f6f8;
}


/* Title */
.page-title {
    text-align: center;
    font-size: 3em;
    color: black !important;
    margin-bottom: 6px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    color: #555 !important;
    margin-bottom: 15px;
}

/* Button */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 22px;
    font-size: 16px;
    border-radius: 6px;
    border: none;
}

/* Card */
.book-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    height: 430px;
    width: 100%;
    max-width: 330px;
    margin: auto;
    margin-bottom: 30px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    
}

/* Image */
.book-card img {
    width: 165px !important;
    height: auto !important;
    object-fit: cover;
}


/* Title */
.book-title {
    font-size: 16px;
    font-weight: 600;
    color: #111;
    margin: 8px 0;
}

/* Author */
.book-author {
    font-size: 15px;
    color: #555;
}

/* Similarity */
.book-score {
    font-size: 14px;
    font-weight: bold;
    margin-top: 6px;
    color: #222;
}

/* Text input wrapper */
[data-testid="stTextInput"] {
    width: 100%;
}

/* Input box */
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #222222 !important;
    border: 1px solid #d0d7de;
    border-radius: 10px;

    height: 52px;
    line-height: 52px;

    padding: 0 16px;               
    font-size: 16px;

    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    box-sizing: border-box !important;
}

/* Placeholder text */
[data-testid="stTextInput"] input::placeholder {
    color: #7a7a7a;
    font-style: italic;
}

/* Focus effect */
[data-testid="stTextInput"] input:focus {
    border-color: #4CAF50;
    box-shadow: 0 0 0 2px rgba(76,175,80,0.2);
    outline: none;
}



</style>
""", unsafe_allow_html=True)



HF_REPO = "Arinjain/book-recommendation-artifacts"

# --------------------------------------------------
# LOAD ARTIFACTS (CACHED)
# --------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_artifacts():
    def load_pkl(filename):
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    books_clean = load_pkl("books_clean.pkl")
    book_embeddings = load_pkl("book_embeddings.pkl")
    title_embeddings = load_pkl("title_embeddings.pkl")
    similarity_matrix = load_pkl("similarity_matrix.pkl")

    snapshot_path = snapshot_download(
        repo_id=HF_REPO,
        allow_patterns="embedding_model/*",
        local_dir_use_symlinks=False
    )

    embedding_model_path = os.path.join(snapshot_path, "embedding_model")
    model = SentenceTransformer(embedding_model_path)

    return books_clean, book_embeddings, title_embeddings, similarity_matrix, model


books_clean, book_embeddings, title_embeddings, similarity_matrix, model = load_artifacts()

# --------------------------------------------------
# RECOMMENDATION FUNCTIONS
# --------------------------------------------------

def recommend_by_embedding_filtered(book_title, top_n=5, min_ratings=500):
    if book_title not in books_clean["title"].values:
        return []

    idx = books_clean.index[books_clean["title"] == book_title][0]

    scores = sorted(
        list(enumerate(similarity_matrix[idx])),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for i, score in scores:
        if i == idx:
            continue
        if books_clean.iloc[i]["ratings_count"] >= min_ratings:
            results.append({
                "title": books_clean.iloc[i]["title"],
                "authors": books_clean.iloc[i]["authors"],
                "image_url": books_clean.iloc[i]["final_image_url"],
                "similarity": round(score, 3)
            })
        if len(results) == top_n:
            break

    return results


def recommend_from_user_text(user_text, top_n=5):
    user_embedding = model.encode(user_text, normalize_embeddings=True)

    scores = cosine_similarity(
        [user_embedding],
        book_embeddings
    )[0]

    top_indices = scores.argsort()[::-1][:top_n]

    return [
        {
            "title": books_clean.iloc[idx]["title"],
            "authors": books_clean.iloc[idx]["authors"],
            "image_url": books_clean.iloc[idx]["final_image_url"],
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
    if not user_input:
        return []

    user_embedding = model.encode(user_input, normalize_embeddings=True)

    title_scores = cosine_similarity(
        [user_embedding],
        title_embeddings
    )[0]

    best_idx = np.argmax(title_scores)
    best_score = title_scores[best_idx]

    if best_score >= title_threshold:
        detected_title = books_clean.iloc[best_idx]["title"]
        return recommend_by_embedding_filtered(
            detected_title, top_n, min_ratings
        )

    return recommend_from_user_text(user_input, top_n)

# --------------------------------------------------
# UI
# --------------------------------------------------

st.markdown("""
<h1 class="page-title">📚 Book Recommendation System</h1>
""", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Search by book title or describe what you want to read</div>", unsafe_allow_html=True)


query = st.text_input(
    label="",
    placeholder="e.g. Twilight, Harry Potter, romantic tragedy..."
)


if st.button("Recommend") and query:
    with st.spinner("Finding best recommendations..."):
        results = hybrid_embedding_recommender(query)

    if results:
        cols = st.columns(4)
    
        for i, book in enumerate(results):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="book-card">
                    <img src="{book['image_url']}" />
                    <div>
                        <div class="book-title">{book['title']}</div>
                        <div class="book-author">{book['authors']}</div>
                    </div>
                    <div class="book-score">
                        Similarity: {book['similarity']}
                    </div>
                </div>
                """, unsafe_allow_html=True)


    else:
        st.warning("No recommendations found.")
