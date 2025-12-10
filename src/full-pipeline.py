#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import lyricsgenius
import nltk
import sentence_transformers
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

artist_name = "INSERT_PREFERRED_ARTIST"
cluster_count = "INSERT_PREFERRED_CLUSTER_NO."
genius_api_token = "INSERT_YOUR_TOKEN"
#ENTER GROQ API TOKEN BELOW ON LINE 160!

# genius api info
genius = lyricsgenius.Genius(
    genius_api_token,
    timeout=15,
    retries=3,
    skip_non_songs=True,
    remove_section_headers=True,
    excluded_terms = ['Remix', 'Version']
)


data = []
scores = []
#scrape_genius_for_StrayKids_songs
def get_jvke(limit=500):
    artist = genius.search_artist(artist_name, max_songs=limit, sort="title")
    for song in artist.songs:
        sentiment = sia.polarity_scores(song.lyrics)
        data.append({
            "Title": song.title,
            "Lyrics": song.lyrics,
            "Score": sentiment['compound']
        })
    time.sleep(1)
    df = pd.DataFrame(data)
    df.to_csv("jvke2.csv", index=False, encoding="utf-8")
    print("Saved:", len(df), "songs")
    return df
get_jvke(360)

df = pd.read_csv("jvke2.csv")

#classify VADER scores
def emotions(score):
    if score > 0.3:
        return 'POSITIVE'
    elif score < -0.3:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'
df['Emotional Scale'] = df['Score'].apply(emotions)
df.to_csv("jvke_song_emotions.csv", index=False)

#assign keywords to VADER polarity scores
label_map = {"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 2}
df["Label"] = df["Emotional Scale"].map(label_map)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Label"])
print("Train label distribution")
print(train_df["Label"].value_counts(),"/n")
print("Validation label distribution")
print(val_df["Label"].value_counts())

import requests

#extract title of songs
def search_song_id(query):
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {genius_api_token}"}
    params = {"q": query}
    r = requests.get(url, headers=headers, params=params).json()
    try:
        result = r["response"]["hits"][0]["result"]
        return result["id"], result["full_title"]
    except:
        return None, None

#extract release date of songs
def fetch_release_date(title):
    song_id, full_title = search_song_id(title)
    if song_id is None:
        return None
    url = f"https://api.genius.com/songs/{song_id}"
    headers = {"Authorization": f"Bearer {genius_api_token}"}
    r = requests.get(url, headers=headers).json()
    try:
        return r["response"]["song"].get("release_date_for_display")
    except:
        return None

df = pd.read_csv("jvke_song_emotions.csv")
release_dates = []
for title in df["Title"]:
    release_dates.append(fetch_release_date(title))
df["Release Date"] = release_dates
df.to_csv("jvke_song_emotions.csv", index=False)
print("Done! New column saved.")


from sentence_transformers import SentenceTransformer

#generate embeddings
embedder = SentenceTransformer("all-mpnet-base-v2")
df["Embedding"] = df["Lyrics"].apply(lambda x: embedder.encode(str(x)).tolist())
df.to_csv("jvke_song_emotions.csv", index=False)
df.head

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#compare songs' vectors' cosine similarity and load it into similarity_matrix
embedding_matrix = np.vstack(df["Embedding"].values)
similarity_matrix = cosine_similarity(embedding_matrix)
similarity_matrix[:5, :5]

from sklearn.cluster import KMeans
#creating clusters by unsupervised learning through K-Means
kmeans = KMeans(cluster_count, random_state = 42)
df["Cluster"] = kmeans.fit_predict(embedding_matrix)
df.to_csv("jvke_song_emotions.csv", index=False)
df[["Title", "Cluster"]]

cluster_samples = {}
for x in df["Cluster"].unique():
    cluster_rows = df[df["Cluster"] == x]["Lyrics"]
    sample_size = min(3, len(cluster_rows))
    sample_texts = df[df["Cluster"] == x]["Lyrics"].sample(sample_size, random_state=42).tolist()
    cluster_samples[x] = sample_texts
cluster_samples

from sklearn.feature_extraction.text import TfidfVectorizer

df["Lyrics"].isna().sum()
df["Lyrics"] = df["Lyrics"].fillna("")
# Get TF-IDF for all lyrics
tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
X_tfidf = tfidf.fit_transform(df["Lyrics"])
terms = tfidf.get_feature_names_out()

# Get top words for each cluster
cluster_keywords = {}
for cluster_id in sorted(df["Cluster"].unique()):
    idx = df[df["Cluster"] == cluster_id].index
    cluster_tfidf = X_tfidf[idx].mean(axis=0).A1
    top_indices = cluster_tfidf.argsort()[::-1][:10]
    top_words = [terms[i] for i in top_indices]
    cluster_keywords[cluster_id] = top_words
cluster_keywords

from groq import Groq

client = Groq(api_key="INSERT_YOUR_TOKEN")
cluster_names = {}
#auto-generate cluster names using Groq-hosted LLM
for c in sorted(df["Cluster"].unique()):
    # Combine the lyrics for this cluster
    combined = "\n\n---\n\n".join(cluster_samples[c])
    # Build the single-cluster prompt
    prompt = f"""
You are an NLP expert. Below is a set of song lyric excerpts from ONE cluster.
Your task:
- Identify one overarching emotional/theme title representing this cluster.
STRICT RULES:
- 3 to 5 words
- Plain text only
- No punctuation except spaces
- No lists
- No numbering
- No quotes
- No markdown
- Cannot have any words in common with the other titles
- Titles cannot be similar in the theme they convey
- Each title must convey a COMPLETELY different emotion/theme
- Output ONLY the theme title (NOTHING else)
CLUSTER {c} LYRICS:
{combined}
Now output exactly one theme title:
"""
    # Send the request for THIS cluster
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    # Extract title
    title = response.choices[0].message.content.strip()
    cluster_names[c] = title
cluster_names
df["Cluster Names"] = df["Cluster"].map(cluster_names)
df.to_csv("jvke_song_emotions.csv", index=False)

from sklearn.manifold import TSNE
import plotly.express as px

embedding_matrix = np.vstack(df["Embedding"].to_numpy())
tsne = TSNE(n_components = 2, random_state = 42, perplexity = 5)
tsne_results = tsne.fit_transform(embedding_matrix)
df["TSNE_1"] = tsne_results[:, 0]
df["TSNE_2"] = tsne_results[:, 1]

#plot TF-IDF vector space with color-coded clusters
fig = px.scatter(
    df,
    x="TSNE_1",
    y="TSNE_2",
    color="Cluster Names",
    hover_data=["Title", "Cluster Names"],
    title="t-SNE Visualization of Song Emotion Clusters",
    height=780,
    width=920
)

#fix title position
fig.update_layout(title_text="Stray Kids Discography Cluster Map (TF-IDF vector projection, color-coded by cluster)", title_x=0.5)
fig.show()

import seaborn as sns
#plot chronological frequency of all clusters
df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
df["Year"] = df["Release Date"].dt.year
timeline = df.groupby(["Year", "Cluster Names"]).size().reset_index(name = "Count")
plt.figure(figsize=(6,6))
sns.lineplot(data=timeline, x = "Year", y = "Count", hue = "Cluster Names", marker = "o")
plt.xlim(left=2018)
plt.xlim(right=2025)
plt.title("Stray Kids Song Topics over Time")
plt.legend(loc="upper right", bbox_to_anchor=(1.8, 1), ncol=1)
plt.show

#plot numerical frequency of all clusters
plt.figure(figsize=(7,5))
df["Emotional Scale"].value_counts().plot(kind="bar")
plt.title("Stray Kids Songs by Predominant Emotion")
plt.show

import matplotlib.pyplot as plt
# Count songs per cluster
cluster_counts = df["Cluster"].value_counts().sort_index()
# Map cluster numbers → LLM-generated names
labels = [cluster_names[c] for c in cluster_counts.index]
plt.figure(figsize=(10, 5))
plt.bar(labels, cluster_counts.values)
plt.title("Song Count per Emotional Cluster")
plt.xlabel("Emotional Cluster")
plt.ylabel("Number of Songs")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
