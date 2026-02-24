# 📚 Mining the Random — Text Mining on the Library of Babel

> *"The Library of Babel contains all possible books... somewhere in it lies every truth, every coherent thought — and also pure chaos."*
> — Jorge Luis Borges

---

## 🧭 Overview

This project applies modern Natural Language Processing (NLP) techniques to randomly extracted documents from [libraryofbabel.info](https://libraryofbabel.info) — a digital implementation of Borges' infinite library, where every book is a random combination of 25 characters (letters, spaces, and punctuation).

The central question: **Does true randomness leave traces? Can chaos generate patterns?**

---

## 🎯 Objective

Perform an exhaustive text mining analysis on random documents from the Library of Babel to:

- Validate the genuine randomness of the texts
- Detect possible emergent patterns or structures
- Explore whether chaos generates *"islands of meaning"*

---

## 🗂️ Dataset

| Parameter | Value |
|-----------|-------|
| Books analyzed | 10 random books |
| Characters per book | ~1,328,800 |
| Total characters | ~13,288,000 |
| Source | libraryofbabel.info |

---

## 🔬 Methodology

The analysis follows a full NLP pipeline:

1. **Tokenization** — Text splitting and token extraction
2. **Stopwords filtering** — Identifying and removing common words
3. **Stemming** — Root extraction analysis
4. **Lemmatization** — Morphological normalization
5. **Frequency Analysis** — Most and least frequent token study
6. **Vectorization** — Bag of Words (BoW) and TF-IDF
7. **Sentiment Analysis** — Emotional valence of random text
8. **Correlation Mapping** — Cross-book pattern concurrence

---

## 📊 Key Findings — The Chaos Is Not Perfect

Despite being genuinely random text with no linguistic meaning, the analysis revealed consistent and surprising biases across all 10 books:

### 🔢 Tokenization Bias
All 10 books tokenized to approximately **~58,000 tokens each**, with average token lengths of the same order of magnitude — a statistically unlikely consistency in truly random text.

### 🚫 Stopwords Bias (~9%)
Even though the texts belong to no real language, the NLTK stopwords library flagged roughly **9% of tokens** across all 10 books consistently. Random character combinations are apparently generating sequences that match known stopword patterns.

### 📝 Lemmatization Emergence
Despite the texts being linguistically meaningless, the lemmatizer was still able to **lemmatize a subset of tokens** in all 10 books — and in similar proportions. The library finds structure where there should be none.

### 🔡 Least Frequent Words: Always 3 Characters
A particularly striking pattern: across all books, the least frequent tokens were **consistently 3 characters long**. This suggests a structural constraint in the randomness itself.

### 🗺️ The Concurrence of Randomness
The **cross-book correlation map** is perhaps the most revealing result — showing that random books share more statistical similarities than expected, visualizing how chaos converges on itself.

---

## 🛠️ Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **Pandas / NumPy** — Data manipulation
- **NLTK** — Tokenization, stopwords, stemming, lemmatization
- **TextBlob / SpaCy** — Sentiment analysis and lemmatization
- **Scikit-learn** — Vectorization (BoW, TF-IDF), LDA
- **Matplotlib / Seaborn** — Visualization
- **WordCloud** — Word frequency visualization

---

## 📁 Repository Structure

```
mining-the-random/
│
├── MINERIA_BIBLIOTECA_BABEL.ipynb   # Main analysis notebook
├── README.md                        # Project documentation
│
├── correlacion_3d_contornos.png     # 3D correlation map
├── correlacion_palabras_tfidf.png   # TF-IDF word correlation
├── correlacion_palabras_top500.png  # Top 500 words correlation
├── wordcloud_babel_raras.png        # Rare words word cloud
├── wordcloud_tokens_filtrados.png   # Filtered tokens word cloud
└── sentimientos_google_cloud.png    # Sentiment analysis chart
```

---

## 🔍 Conclusions

- **Chaos is not perfect** — even "random" text shows measurable statistical biases
- **Patterns emerge** — pseudo-words with emotional valence appear within millions of combinations
- **True randomness is hard** — it seems nearly impossible to generate something truly random that carries no structure whatsoever
- **The Library converges** — correlation maps reveal that random books share unexpected statistical DNA

---

## 👤 Author

**Diego Evans Pérez Paz**
Project developed for the *Complejidad y Caos* course — January 2026

