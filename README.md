# Cluster Beats
> NLP-powered analysis of public music discographies
- Get your Genius API key: https://docs.genius.com/#/getting-started-h1
- Get your Groq API key: https://console.groq.com/keys

## Motivation
Lyrics change. Artists grow. But there's no easy way to track how an artist really evolves throughout their career. Traditional sentiment analysis tools oversimplify themes into "positive", "negative", and "neutral", offering no deeper look into the artist's raw thematic journey. That's why I made Cluster Beats.

Cluster Beats gives us truly meaningful snapshots of an artist's discography, grouping songs by emotional themes through vectorization and unsupervised learning, before finally visualizing these themes and how they change/shift over time. This end-to-end NLP pipeline is my way of combining my love for both programming and music, and helping others do the same. Music isn't simply positive and negative, and our tools to analyze music shouldn't be, either.

## How it works (pipeline overview)
Cluster Beats is a customizable end-to-end NLP pipeline that first scrapes song, artist, and discography data from [genius.com](https://genius.com/) using the **Genius API** and **lyricsgenius**. After normalizing our text data, VADER sentiment analysis is initiated to provide a baseline polarity score for each extracted song. Cluster Beats then converts the lyrics into numerical embeddings using TF-IDF vectorization before applying K-Means to those very vectors to identify shared emotional themes and cluster them accordingly. Following this, a [Groq LLama model](https://console.groq.com/keys) is used to generate labels for each cluster, and the resulting trends are displayed in four visualizations. Cluster Beats can me modified to analyze any music artist, with any number songs, across any time period, making it extremely accessible to all music lovers. Programming experience not required :)

## Demo
For demonstration, I've used Cluster Beats to analyze the K-Pop boyband, Stray Kids' entire discography from 2018-2025, to provide interested users a clear example how Cluster Beats works to analyze your favorite artist. (Click [here](https://github.com/steawinfdox-png/Cluster-Beats/tree/main/notebooks) for more)

### Color-coded 2D PCA Cluster Map of TF-IDF Vectors

   <img width="1225" height="910" alt="image" src="https://github.com/user-attachments/assets/643de493-c9be-4d03-9fcd-c745167f3372" />
  
### Emotional Timeline

   <img width="1131" height="679" alt="Timeline" src="https://github.com/user-attachments/assets/bc318f10-6e62-4542-a824-54e7041873e8" />

### Theme Frequency Chart

   <img width="1304" height="630" alt="Screenshot (189)" src="https://github.com/user-attachments/assets/7bc1bb72-526e-4e48-831d-5f9048eaa701" />

### VADER Sentiment Analysis

   <img width="597" height="526" alt="Pie" src="https://github.com/user-attachments/assets/7ad6047a-cb4b-4d07-b9a5-f09cd6096535" />

## Tech Stack

**Languages + Tools**
* Python (data analysis, visualization, NLP, clustering)
* JupyterLab (interactive dev.)

**Data Extraction**

* Genius API (lyric scraping, song title and release date extraction)
* Requests (API communication)

**Natural Language Processing**

* NLTK (song tokenization)
* scikit-learn (TF-IDF vectorization, K-Means unsupervised clustering)
* VADER (sentiment analysis via polarity scoring)

**LLM's**

* Groq API (automated cluster naming)
* Llama-3.3-70B (label generation)

**Data Visualization**

* Matplotlib (Emotional timeline)
* Seaborn (Trend visuals)
* Plotly (2D PCA cluster distribution)

## Using Cluster Beats on your device
1. Install dependencies
   > pip install -r requirements.txt
2. Open src/FULL_PIPELINE.py
3. Change "artist_name" to your preferred music artist, as well as customize other variables (note: you need to get tokens for Genius API and Groq API)
4. Run pipeline in JupyterLab
   > File --> New --> Terminal
   > Enter and run "python main.py"

📦 **Requirements**

1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. seaborn
6. nltk
7. python-dotenv
8. requests
9. groq

## Conclusion
This project demonstrates how custom NLP methods (TF-IDF vectorization, clustering, and keyword-based emotion classification) can reveal deeper, more nuanced emotional patterns across Stray Kids’ discography than traditional sentiment tools. While VADER provides a broad positive/negative/neutral sentiment score, it often oversimplifies complex emotional tones, especially in music where themes like empowerment, vulnerability, chaos, longing, and ambition coexist within similar songs and reduces them to simply "positive", "negative", and "neutral" keywords. In contrast, the project’s custom emotion/theme pipeline captures these multilayered dynamics, producing a detailed emotional timeline and cluster map that more accurately reflects the group’s artistic evolution and versatility. Together, these insights highlight how domain-specific NLP can uncover emotional narratives that general-purpose sentiment models fail to detect, making this approach far more effective for analyzing lyrical content than traditional sentiment analysis software.

## Journal
I initially attempted to use an OpenAI-hosted LLM to auto-generate cluster names, but I ran out of free credits and my requests greatly exceeded the hourly limit, leading to various, prolonged errors in my code. After several attempts to remedy the issue, I instead opted to use the simpler-to-integrate Groq-hosted LLM.

# Contact Me!
If you've got any suggestions or recommendations, feel free to email me at [steawinfdox@gmail.com](mailto:steawinfdox@gmail.com)!

## Resources Used
* JupyterLab
* Python
* OpenAI
