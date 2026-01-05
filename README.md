# Cluster Beats
> co

## Motivation
Lyrics change. Artists grow. But there's no easy way to track how an artist really evolves throughout their career. Traditional sentiment analysis tools oversimplify themes into "positive", "negative", and "neutral", offering no deeper look into the artist's raw thematic journey. That's why I made Cluster Beats.
Cluster Beats gives us truly meaningful and comprehensive snapshots of a music artist's full discography and groups songs by related themes through unsupervised learning, before finally visualizing these themes and how they change and shift over time. This project is my way of combining my love for both programming and music, and helping others do the same. Music isn't simply positive and negative, and our tools to analyze music shouldn't be, either.

## 🎧 Abstract
🧠 Cluster Beats is a customizable end-to-end NLP pipeline that analyzes any and all music artists' discographies (scraped from Genius API) by clustering lyrics from all of their songs into emotional themes using TF-IDF vectorization and unsupervised learning through K-Means. Next, Cluster Beats auto-generates interpretable labels for each discography cluster using a Groq-hosted LLM, builds a emotional timeline over the artist's entire career (exact number of years can be changed), and visualizes thematic trends and patterns using Matplotlib, Plotly, and Seaborn. In addition to clustering, Cluster Beats employs the use of VADER sentiment analysis to provide baseline polarity scores for each song, allowing comparison between more traditional sentiment scoring and unsupervised learning.

🎤 For demonstration, I've used Cluster Beats to analyze the K-Pop boyband, Stray Kids' entire discography from 2018-2025, to provide interested users a clear example how Cluster Beats works to analyze your favorite artist. (Additional demos are in the notebooks/ file)

✨ **Cluster Beats is completely customizable and accessible to all music lovers (programming experience not required!)**

⚙️ **How to Use Cluster Beats**

1. Install dependencies
   a. pip install -r requirements.txt
2. Open src/FULL_PIPELINE.py
3. Change "artist_name" to your preferred music artist, as well as customize other variables (note: you need to get tokens for Genius API and Groq API)
4. Run pipeline in JupyterLab
   a. File --> New --> Terminal
   b. Enter and run "python main.py"

🔬 **Pipeline Overview**

1. 🎼 Scrape Song Metadata & Lyrics

* Pull complete music artist discography using the Genius API
* Store song titles, release dates, and full lyrics for analysis

2. 🧹 Preprocess Lyrics

* Clean and normalize text (lowercase, remove punctuation, remove stopwords)
* Filter out metadata brackets

3. 😊 Sentiment Pre-Analysis (VADER)

* Generate compound polarity score for each song
* Classify into positive, neutral, or negative to visualize mood trends as supportive supplemental data

4. 🔢 Vectorize Lyrics (TF-IDF)

* Convert lyrics into numeric embeddings using TF-IDF
* Extract key linguistic features for clustering

5. 🔀 Unsupervised Clustering (K-Means)

* Applied K-Means to TF-IDF vectors to identify latent emotional themes
* Evaluated SSE/Elbow method to select k cluster count

6. 🏷️ Cluster Labeling with Groq LLM

* Sent sample lyrics from each cluster to a Groq Llama model
* Generated concise 3–5 word theme labels
* Mapped each track to its theme

7. 📊 Visualization

* Projected high-dimensional vectors into 2D (t-SNE) for visual mapping
* Created cluster frequency charts and an emotional timeline across the 7-year discography using Matplotlib, Plotly, and Seaborn

🧰 **Tech Stack**
🧪 Languages + Tools

* Python (data analysis, visualization, NLP, clustering)
* JupyterLab (interactive dev.)

📡 Data Extraction

* Genius API (lyric scraping, song title and release date extraction)
* Requests (API communication)

🧠 Natural Language Processing

* NLTK (song tokenization)
* scikit-learn (TF-IDF vectorization, K-Means unsupervised clustering)
* VADER (sentiment analysis via polarity scoring)

🤖 LLM's

* Groq API (automated cluster naming)
* Llama-3.3-70B (label generation)

📈 Data Visualization

* Matplotlib (Emotional timeline)
* Seaborn (Trend visuals)
* Plotly (2D PCA cluster distribution)

📦 **## Requirements**

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* nltk
* python-dotenv
* requests
* groq

🌟 **Key Features**

* Full Discography Scraper using the Genius API to automatically download all lyrics and metadata.
* End-to-end NLP Pipeline including preprocessing, sentiment scoring, vectorization, clustering, and labeling.
* Hybrid Emotion Analysis combining traditional sentiment (VADER) with unsupervised clustering (TF-IDF + K-Means).
* LLM-Generated Cluster Names powered by Groq’s LLaMA 3.3–70B for interpretable emotional categories.
* Temporal Analysis showing how Stray Kids’ lyrical emotions evolved across their 7-year career.
* Clear Visualizations including sentiment trends, cluster timelines, and distribution plots.

📊 **Results**

1. 🎶 Emotional Clusters

* The K-Means model (k=5) produced clusters that were semantically consistent when inspected qualitatively
* Groq-hosted LLM cluster names closely matched the lyrical tone and vocabulary of each respective cluster:

  * Unstoppable Inner Strength
  * Self-Empowerment through Adversity
  * Youthful Energy Explosion
  * Empowered Rising Spirit
  * Longing and Heartache

2. 📉 Sentiment Distribution

* VADER sentiment analysis showed dominance of positive polarity scored songs
* Negative sentiment correlated with songs with lyrics about personal struggle and isolation

3. ⏳ Temporal Analysis

* Throughout Stray Kids' entire 7-year career, songs with cluster names with high-energy ("Empowered Rising Spirit" and "Self Empowerment through Adversity") consistently dominated annual theme rankings
* Since 2021-2022, however, has "Self Empowerment through Adversity" dropped significantly in frequency, leaving songs in the "Empowered Rising Spirit" cluster to be the No. 1 cluster theme for Stray Kids

4. 📐 Cluster Geometry

* All clusters were relatively distanced from each other in the 2D PCA, suggesting the embeddings creating meaningful variation
* Such a uniform distribution implies a strong thematic cohesion across Stray Kids' discography even with completely different thematic classifications

🖼️ **Visualizations**
📎 Attached are several visualizations that help interpret the emotional structure and trends of Stray Kids' 7-year discography

1. 🎨 [Color-coded 2D PCA Cluster Map of TF-IDF Vectors]
   (visualizes how songs associate together based on lyrical similarity and shows clear separation between emotional themes)

   <img width="1225" height="910" alt="image" src="https://github.com/user-attachments/assets/643de493-c9be-4d03-9fcd-c745167f3372" />

2. 📈 [Emotional Timeline]
   (line-graph visualization highlighting how emotional cluster themes rise and fall in song frequency over Stray Kids' 7-year career)

   <img width="1131" height="679" alt="Timeline" src="https://github.com/user-attachments/assets/bc318f10-6e62-4542-a824-54e7041873e8" />

3. 📊 [Theme Frequency Chart]
   (shows actual number-based freqency of all clusters from entire discography)

   <img width="1304" height="630" alt="Screenshot (189)" src="https://github.com/user-attachments/assets/7bc1bb72-526e-4e48-831d-5f9048eaa701" />

4. 🧪 [VADER Sentiment Analysis]
   (traditional pre-clustering pos-neg-neu classifications)

   <img width="597" height="526" alt="Pie" src="https://github.com/user-attachments/assets/7ad6047a-cb4b-4d07-b9a5-f09cd6096535" />

🧾 **Conclusion**
📌 This project demonstrates how custom NLP methods (TF-IDF vectorization, clustering, and keyword-based emotion classification) can reveal deeper, more nuanced emotional patterns across Stray Kids’ discography than traditional sentiment tools. While VADER provides a broad positive/negative/neutral sentiment score, it often oversimplifies complex emotional tones, especially in music where themes like empowerment, vulnerability, chaos, longing, and ambition coexist within similar songs and reduces them to simply "positive", "negative", and "neutral" keywords. In contrast, the project’s custom emotion/theme pipeline captures these multilayered dynamics, producing a detailed emotional timeline and cluster map that more accurately reflects the group’s artistic evolution and versatility. Together, these insights highlight how domain-specific NLP can uncover emotional narratives that general-purpose sentiment models fail to detect, making this approach far more effective for analyzing lyrical content than traditional sentiment analysis software.

📝 **Journal**
I initially attempted to use an OpenAI-hosted LLM to auto-generate cluster names, but I ran out of free credits and my requests greatly exceeded the hourly limit, leading to various, prolonged errors in my code. After several attempts to remedy the issue, I instead opted to use the simpler-to-integrate Groq-hosted LLM.

📬 **Contact Me!**
If you've got any suggestions or recommendations, feel free to email me at [steawinfdox@gmail.com](mailto:steawinfdox@gmail.com)!

📚 **Resources Used**

* JupyterLab
* Python
* OpenAI
