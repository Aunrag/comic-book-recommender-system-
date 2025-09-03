import pandas as pd
import ast
import nltk
from sklearn.neighbors import NearestNeighbors

df=pd.read_csv("comic_data.csv")

df.drop_duplicates(inplace=True)

df.drop(['rating','year'],axis=1,inplace=True)

df.drop('cover',axis=1,inplace=True)

df.dropna(inplace=True)

def convert(x):
    a=ast.literal_eval(x)
    return a

df["tags"]=df['tags'].apply(convert)

df['description']=df['description'].apply(lambda x:x.split())

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(x):
    l=[]
    for i in (x):
        a=ps.stem(i)
        l.append(a)
    return l

a=df.iloc[0]['tags']
b=stem(a)

df['description']=df['description'].apply(stem)

df['tags']=df['tags'].apply(lambda x:[i.replace(" ","")for i in x])

df['tags']=df['description']+df['tags']

df.drop(["description"],axis=1,inplace=True)

df['tags']=df['tags'].apply(lambda x:' '.join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000,stop_words="english")

vector=cv.fit_transform(df['tags'])



model=NearestNeighbors(n_neighbors=10, metric='cosine')

model.fit(vector)

distance,indices=model.kneighbors(vector[2])


df['title_lower'] = df['title'].str.lower()
def recommend(x):
    x_lower = x.lower()
    if x_lower not in df['title_lower'].values:
        print(f"'{x}' is not in the dataset.")
        return
    i=df[df['title_lower']==x_lower].index
    distance,indices=model.kneighbors(vector[i])
    print(f"similar comics to [{df.iloc[i]['title']}] are:")
    for j in indices[0]:
        if j!=i:
            print("recommended comic: ",df.iloc[j]['title'])
    print()



