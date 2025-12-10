import matplotlib as plt
#plot numerical frequency of all clusters
plt.figure(figsize=(7,5))
df["Emotional Scale"].value_counts().plot(kind="pie")
plt.title("Stray Kids Songs by Predominant Emotion")
plt.show
