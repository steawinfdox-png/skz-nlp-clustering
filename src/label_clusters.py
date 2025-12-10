from groq import Groq

cluster_samples = {}
for x in df["Cluster"].unique():
    cluster_rows = df[df["Cluster"] == x]["Lyrics"]
    sample_size = min(3, len(cluster_rows))
    sample_texts = df[df["Cluster"] == x]["Lyrics"].sample(sample_size, random_state=42).tolist()
    cluster_samples[x] = sample_texts

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
