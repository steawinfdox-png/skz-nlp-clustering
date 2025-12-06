# Unsupervised Emotional Clustering and Visualization of Stray Kids' Discography
An end-to-end NLP pipeline that analyzes the music artist, Stray Kids' entire 7-year Korean discography (scraped from Genius API) by clustering Korean/English lyrics into emotional themes using TF-IDF vectorization and unsupervised learning through K-Means. This project auto-generates interpretable cluster labels using a Groq-hosted LLM, builds a emotional timeline over the group's career from 2018-2025, and visualizes thematic trends and patterns using Matplotlib and Seaborn. Prior to clustering, I used VADER for sentiment analysis to provide baseline polarity scores scores for each song, allowing comparison between traditional sentiment scoring and unsupervised clustering. This project's code can also be analyzed any music artist's emotional trends, making it customizable and accessible to anyone who simply enjoys music, even if they don't know programming.

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
- VADER (sentiment analysis via polarity scoring)
  
LLM's
- Groq API (automated cluster naming)
- Llama-3.3-70B (label generation)
  
Data Visualization
- Matplotlib (Emotional timeline)
- Seaborn (Trend visuals)
- Plotly (Scatterplot distribution)

# Pipeline Overview

1. Scrape Song Metadata & Lyrics
- Pull complete music artist discography using the Genius API
- Store song titles, release dates, and full lyrics for analysis

2. Preprocess Lyrics
- Clean and normalize text (lowercase, remove punctuation, remove stopwords)
- Filter out metadata brackets

3. Sentiment Pre-Analysis (VADER)
- Generate compound polarity score for each song
- Classify into positive, neutral, or negative to visualize mood trends

4. Vectorize Lyrics (TF-IDF)
- Convert lyrics into numeric embeddings using TF-IDF
- Extract key linguistic features for clustering

5. Unsupervised Clustering (K-Means)
- Group songs into 5 emotional clusters based purely on lyrical similarity
- Identify meaningful structure without labels

6. Cluster Labeling with Groq LLM
- Summarize each cluster’s themes using LLaMA-3.3-70B via Groq API
- Produce clean 3–5 word category names for each emotional cluster

7. Visualization
- Plot emotional cluster distribution across 7-year timeline
- Visualize sentiment and thematic trends using Matplotlib, Seaborn

# Key Features
- Full Discography Scraper using the Genius API to automatically download all lyrics and metadata.
- End-to-end NLP Pipeline including preprocessing, sentiment scoring, vectorization, clustering, and labeling.
- Hybrid Emotion Analysis combining traditional sentiment (VADER) with unsupervised clustering (TF-IDF + K-Means).
- LLM-Generated Cluster Names powered by Groq’s LLaMA 3.3–70B for interpretable emotional categories.
- Temporal Analysis showing how Stray Kids’ lyrical emotions evolved across their 7-year career.
- Clear Visualizations including sentiment trends, cluster timelines, and distribution plots.
