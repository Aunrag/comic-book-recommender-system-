## Manga and Manhwa Recommender System

### Project Overview

This project is a content-based recommender system tailored for manga and manhwa enthusiasts. It utilizes metadata like tags and descriptions to find similar comics using natural language processing and unsupervised machine learning techniques. The core idea is to provide top-k recommendations based on textual similarity using cosine distance.

---

###  Project Workflow

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

### ML / NLP Techniques Used

* **Tokenization & Stemming**: Using NLTK's `PorterStemmer` for feature normalization.
* **Count Vectorization**: For converting text to numerical vectors.
* **Nearest Neighbors Algorithm**: Scikit-learn implementation to find similar comics based on cosine distance.

---

### Dependencies
```bash
pip install pandas scikit-learn nltk
```
```python
import nltk
nltk.download('punkt')
```

---

### How to Run

1. Clone the repository or download the notebook.
2. Place your `comic_data.csv` file in the same directory.
3. Run the notebook cells sequentially.
4. Use the input cell to test your recommendations by passing a comic title.

---

### Example Output

```python
    recommend('Omniscient Reader')
    output
        similar comics to [9    Omniscient Reader
        Name: title, dtype: object] are:
        recommended comic:  Omniscient Reader (Novel)
        recommended comic:  Past Lives of the Thunder God
        recommended comic:  Stairs
        recommended comic:  The Fighting Monster
        recommended comic:  Shaman (Won-Pyo HONG)
        recommended comic:  Shi ni Modori, Subete wo Sukuu Tame ni Saikyou e to Itaru
        recommended comic:  I Become A Villain In My Novel
        recommended comic:  My Lady, Please Become My Heroine!
        recommended comic:  Ouroboros Record: Circus of Oubeniel (Light Novel   )
```

---

