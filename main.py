from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from recommender import recommend_comics

app = FastAPI()

# Allow frontend HTML to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComicRequest(BaseModel):
    comic_name: str

@app.post("/recommend")
def recommend(request: ComicRequest):
    recs = recommend_comics(request.comic_name)
    return {"recommendations": recs}

