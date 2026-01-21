import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def parse_episode_info(name: str):
    pattern = r'Series (\d{2}) Episode (\d{2}) – (.+)'

    match = re.match(pattern, name)
    if match:
        return match.groups()

    return "Unknown", "Unknown", "Unknown"



# Initialize Vector DB (Chroma) and Embedding Model
# We use 'all-MiniLM-L6-v2' because it's fast and runs well on local CPUs.
print("Initializing model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# Delete collection if it exists to start fresh, then create it
try:
    chroma_client.delete_collection(name="tbbt_transcripts")
except:
    pass
collection = chroma_client.create_collection(name="tbbt_transcripts")

def ingest_data():
    print("Loading dataset.csv...")
    if not os.path.exists(CSV_PATH):
        print("Error: dataset.csv not found in the backend folder.")
        return

    # Load CSV. Kaggle TBBT datasets usually have 'Person' and 'Dialogue' columns.
    # We will inspect the columns or assume standard names.
    df = pd.read_csv(CSV_PATH, encoding="latin1")
    df["episode_name"] = df["episode_name"].str.replace('\xa0', ' ')

    # Standardize column names if necessary (adjust this based on your actual CSV)
    # Common Kaggle TBBT headers: 'person_scene', 'dialogue' OR 'Speaker', 'Text'
    # Let's clean up and rename for consistency
    df.columns = [c.lower() for c in df.columns]
    
    # Map your CSV columns to our needs. 
    # ADJUST THESE IF YOUR CSV HEADERS ARE DIFFERENT
    if 'person_scene' in df.columns: # One common version of the dataset
        df['speaker'] = df['person_scene']
    elif 'person' in df.columns:
        df['speaker'] = df['person']
        
    if 'dialogue' in df.columns:
        df['text'] = df['dialogue']
    
    # Filter out empty rows
    df = df.dropna(subset=['text'])

    batch_size = 100
    docs = []
    ids = []
    metadatas = []
    
    print(f"Ingesting {len(df)} rows...")

    for index, row in df.iterrows():
        # Create the content to be embedded. 
        # Including the speaker helps the AI understand context (e.g., "Sheldon said...")
        text_content = f"{row.get('speaker', 'Unknown')}: {row.get('text', '')}"
        
        docs.append(text_content)
        ids.append(f"id_{index}")
        
        # Metadata is what we get back *after* the search finds a match
        season, episode, episode_title = parse_episode_info(
            row.get("episode_name", "")
        )

        metadatas.append({
            "speaker": str(row.get('speaker', 'Unknown')),
            "season": season,
            "episode": episode,
            "episode_title": episode_title,
            "context": text_content[:100] + "..."
        })

        if len(docs) >= batch_size:
            embeddings = model.encode(docs).tolist()
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            docs = []
            ids = []
            metadatas = []
            print(f"Indexed {index + 1} rows...")
    if docs:
        embeddings = model.encode(docs).tolist()
        collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Indexed final {len(docs)} rows.")
    print("Ingestion Complete! Vector DB is ready.")

if __name__ == "__main__":
    ingest_data()
