from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assume you have these lists
words = [
    "Python",
    "programming",
    "data",
    "analysis",
    "web",
    "development",
    "machine",
    "learning",
]
frequencies = [5, 3, 4, 3, 2, 2, 3, 4]

# Combine the words and frequencies into a dictionary
freq_dict = dict(zip(words, frequencies))

# Create the wordcloud object
wordcloud = WordCloud(
    width=480, height=480, prefer_horizontal=1.0, margin=0, background_color="white"
).generate_from_frequencies(freq_dict)

# Display the generated image:
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
