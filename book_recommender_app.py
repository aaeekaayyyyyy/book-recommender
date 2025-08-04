import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import os

openai_key = os.getenv('OPENAI_API_KEY')
hf_key = os.getenv('HUGGINGFACE_API_KEY')


from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Load your books CSV
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Prepare your semantic search
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0.0000000000000000000000000001, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    if category != "All" and category:
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)
    return book_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# Streamlit UI
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("Book Recommender")

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

with col1:
    user_query = st.text_input("Please enter a description of a book:", placeholder="e.g., A story about forgiveness")
with col2:
    category = st.selectbox("Select a category:", categories, index=0)
with col3:
    tone = st.selectbox("Select an emotional tone:", tones, index=0)
with col4:
    submit = st.button("Find recommendations")

st.markdown("## Recommendations")

if submit and user_query:
    results = recommend_books(user_query, category, tone)
    n_cols = 8
    n_rows = 2
    # Arrange in a grid with images and captions
    for i in range(n_rows):
        row = st.columns(n_cols)
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(results):
                image_url, caption = results[idx]
                with row[j]:
                    st.image(image_url, caption=caption, use_container_width=True)
            else:
                with row[j]:
                    st.empty()
