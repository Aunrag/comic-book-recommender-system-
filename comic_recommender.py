import pandas as pd
import joblib                     # or pickle
# -------- load artefacts ----------
df       = pd.read_csv("comic_data.csv")          # your titles
vector   = joblib.load("vector_matrix.pkl")   # the TF‑IDF / CountVectorizer matrix
model    = joblib.load("knn_model.pkl")       # the fitted NearestNeighbors model

df["title_lower"] = df["title"].str.lower()

def recommend(title: str, top_k: int = 10) -> list[str]:
    """Return up to top_k similar comics for the given title."""
    t = title.lower()
    if t not in df["title_lower"].values:
        return []                       # not found
    idx         = df.index[df["title_lower"] == t][0]
    _, indices  = model.kneighbors(vector[idx])
    rec_titles  = [
        df.iloc[j]["title"]
        for j in indices[0] if j != idx            # skip the query itself
    ]
    return rec_titles[:top_k]
