# Unsupervised Emotional Clustering and Visualization of Stray Kids' Discography
An end-to-end NLP pipeline that analyzes the music artist, Stray Kids' entire 7-year Korean discography (scraped from Genius API) by clustering Korean/English lyrics into emotional themes using TF-IDF vectorization and unsupervised learning through K-Means. This project auto-generates interpretable cluster labels using a Groq-hosted LLM, builds a emotional timeline over the group's career from 2018-2025, and visualizes thematic trends and patterns using Matplotlib and Seaborn. Prior to clustering, I used VADER for sentiment analysis to provide a baselien for pos/neg/neu scores for each song, allowing comparison between traditional sentiment scoring and unsupervised clustering.

# Tech Stack
Languages + Tools
- Python (data analysis, visualization, NLP, clustering)
- JupyterLab (interactive dev.)
  
Data Extraction
- Genius API (lyric scraping, song title and release date extraction)
- Requests (API communication)
  
Natural Language Processing
- NLTK (song tokenization)
- scikit-learn (TF-IDF vectorization, K-Means unsupervised clustering)
  
LLM's
- Groq API (automated cluster naming)
- Llama-3.3-70B (label generation)
  
Data Visualization
- Matplotlib (Emotional timeline)
- Seaborn (Trend visuals)
- Plotly (Scatterplot distribution)g
