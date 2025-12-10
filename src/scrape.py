genius = lyricsgenius.Genius(
    #INITIALIZE GENIUS API CLIENT
    #Copy-paste this link to set up a Genius API client and get a token: https://genius.com/developers
    #Copy-paste this link to for Genius API documentation for your own projects: https://docs.genius.com/#/getting-started-h1 
    genius_api_token,
    timeout=15,
    retries=3,
    skip_non_songs=True,
    remove_section_headers=True,
    excluded_terms = ['Remix', 'Version']
)


data = []
scores = []
def get_jvke(limit=500):
    #SCRAPE GENIUS FOR SONGS BY THE ARTIST
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

#CHANGE THIS NUMBER DEPENDING ON HOW MANY SONGS YOU WANT IN YOUR DATA ANALYSIS
get_jvke(360)
