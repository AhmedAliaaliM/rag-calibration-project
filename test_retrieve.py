import chromadb
from sentence_transformers import SentenceTransformer

c = chromadb.PersistentClient(path=r'D:\project\rag-project\chroma_experiments')
col = c.get_collection('v2_sentence_300')
print('Collection count:', col.count())

model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode('What is a data lake?').tolist()

results = col.query(query_embeddings=[query_embedding], n_results=5)
print('Number of results:', len(results['documents'][0]))
print('Sources:', [m['source'] for m in results['metadatas'][0]])