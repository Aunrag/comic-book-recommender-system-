## 📘 Manga and Manhwa Recommender System

### 🧠 Project Overview

This project is a content-based recommender system tailored for manga and manhwa enthusiasts. It utilizes metadata like tags and descriptions to find similar comics using natural language processing and unsupervised machine learning techniques. The core idea is to provide top-k recommendations based on textual similarity using cosine distance.

---

### ⚙️ Project Workflow

1. **Data Cleaning and Preprocessing**

   * Removal of null and duplicate records.
   * Parsing nested string fields using `ast.literal_eval`.
   * Basic NLP on descriptions and tags: tokenization, stemming, and whitespace removal.

2. **Text Representation**

   * Combined `tags` and `description` fields into a single `tags` field.
   * Converted text into vector format using `CountVectorizer`.

3. **Modeling**

   * Used `NearestNeighbors` with cosine similarity to find similar titles.
   * Returned the top-N nearest neighbors as recommendations.

4. **Recommendation Logic**

   * User inputs a title, and the system returns top similar comics based on tag similarity.

---

### 🧪 ML / NLP Techniques Used

* **Tokenization & Stemming**: Using NLTK's `PorterStemmer` for feature normalization.
* **Count Vectorization**: For converting text to numerical vectors.
* **Nearest Neighbors Algorithm**: Scikit-learn implementation to find similar comics based on cosine distance.

---

### 📦 Dependencies

Make sure to install the following dependencies before running the notebook:

```bash
pip install pandas scikit-learn nltk
```

Also, don’t forget to download NLTK resources if not already available:

```python
import nltk
nltk.download('punkt')
```

---

### ▶️ How to Run

1. Clone the repository or download the notebook.
2. Place your `comic_data.csv` file in the same directory.
3. Run the notebook cells sequentially.
4. Use the input cell to test your recommendations by passing a comic title.

---

### 🧾 Example Output

```python
recommend("Solo Leveling")
# Output:
# ['The Beginning After the End', 'Tomb Raider King', 'Ranker Who Lives A Second Time', ...]
```

---

## ✅ Code Review Summary

### ✔️ Strengths

* Good use of preprocessing (e.g. stemming, removing noise).
* Uses a simple but effective ML model for recommendations.
* Well-structured transformation pipeline.
* Clean handling of missing and duplicate data.

### ❗ Suggestions

* Add exception handling if the input title is not in the dataset.
* Consider using TF-IDF instead of CountVectorizer for more nuanced similarity.
* Visualize similarity using a heatmap or dimensionality reduction (e.g. PCA, t-SNE).
* Modularize code into functions or classes for scalability.

Would you like me to convert this into a `README.md` file or a downloadable `.txt`/`.md` version?
