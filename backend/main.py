from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

app = FastAPI()

# Enable CORS so your React app (localhost:3000) can talk to this Python app (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources once when server starts
print("Loading Vector DB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name="tbbt_transcripts")
print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/search")
def search(q: str):
    print(f"Searching for: {q}")
    
    # 1. Turn user query into numbers
    query_vector = model.encode([q]).tolist()
    
    # 2. Find nearest matches in DB
    results = collection.query(
        query_embeddings=query_vector,
        n_results=10 # Fetch top 10 to have variety
    )
    
    formatted_results = []
    
    if results['documents']:
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            
            # Distance logic: 0 is identical, 1 is very different.
            # Convert to a confidence score.
            confidence = 1.0 - (dist / 2) 
            
            formatted_results.append({
                "season": meta.get('season', '?'),
                "episode": meta.get('episode', '?'),
                "episodeTitle": f"S{meta.get('season')}E{meta.get('episode')}",
                "timestamp": "N/A", # Dataset usually doesn't have timestamps
                "speaker": meta.get('speaker', 'Unknown'),
                "context": meta.get('context', ''),
                "fullQuote": results['documents'][0][i],
                "confidence": confidence
            })
            
    return formatted_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
