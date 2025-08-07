
LLM-Powered Book Discovery Engine: From Embeddings to Emotions
LLM-Powered Book Discovery Engine: From Embeddings to Emotions
I built a semantic book recommendation app leveraging modern NLP and LLMs to suggest books based on the actual meaning and sentiment of descriptions—not just keywords or genres.

link: https://book-recommended.streamlit.app/

What’s under the hood:

- Cleaned a 7K+ book dataset: standardized messy metadata, handled missing fields, created unified title+description, and mapped cleaner genres.

- Used OpenAI’s embeddings to turn every book description into a high-dimensional vector representing meaning.

- Indexed all vectors in ChromaDB for near-instant similarity search—so users can enter any natural language query (“funny dystopian adventure,” “heartwarming memoir”) and get semantically matched book recs.

- Added zero-shot classification with HuggingFace transformers to auto-categorize fiction/nonfiction and fine-tuned RoBERTa models to infer emotional tone (joy, sadness, suspense, etc.), making “vibe-based” filtering possible.

- Modular, API-first pipeline for easy upgrades and add-ons.

- Streamlit powered frontend: users search by description, filter by genre/emotion, and browse recommendations in a visual gallery.

Engineering takeaways:
Semantic embeddings and vector DBs enable more nuanced, “vibe-matching” recommendations than keyword search. Zero-shot and fine-tuned models make robust filtering simple, even without labeled data. Persisting embeddings slashes costs and boosts UX.
