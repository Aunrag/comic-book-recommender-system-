## Comic Book Recommender System

### Project Overview

This project is a content-based recommender system tailored for manga and manhwa enthusiasts. It utilizes metadata like tags and descriptions to find similar comics using natural language processing and unsupervised machine learning techniques. The core idea is to provide top-k recommendations based on textual similarity using cosine distance.

---

### Project Workflow

1. **Data Cleaning and Preprocessing**
   - Removal of null and duplicate records.
   - Parsing nested string fields using `ast.literal_eval`.
   - Basic NLP on descriptions and tags: tokenization, stemming, and whitespace removal.

2. **Text Representation**
   - Combined `tags` and `description` fields into a single `tags` field.
   - Converted text into vector format using `CountVectorizer`.

3. **Modeling**
   - Used `NearestNeighbors` with cosine similarity to find similar titles.
   - Returned the top-N nearest neighbors as recommendations.

4. **Recommendation Logic**
   - User inputs a title, and the system returns top similar comics based on tag similarity.

---

### ML / NLP Techniques Used

- **Tokenization & Stemming**: Using NLTK's `PorterStemmer` for feature normalization.
- **Count Vectorization**: For converting text to numerical vectors.
- **Nearest Neighbors Algorithm**: Scikit-learn implementation to find similar comics based on cosine distance.

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

### Example Output

```python
recommend('Omniscient Reader')
```
   Output
```
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

---

### Flask Web App Integration

A simple web interface is added using **Flask**, allowing users to input a comic title and receive recommendations directly in the browser.

#### 📷 Web App UI Screenshot

![Web App Screenshot](static/flask_ui.jpg)


#### 🔧 How to Run the Flask App

1. Clone this repository.

2. Ensure you have `comic_data.csv` in the root directory.

3. Install dependencies:
    ```bash
    pip install pandas scikit-learn nltk flask
    ```

4. Download NLTK tokenizer if not already done:
    ```python
    import nltk
    nltk.download('punkt')
    ```

5. Run the Flask app:
    ```bash
    python app.py
    ```

6. Open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---


