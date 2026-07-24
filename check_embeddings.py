import chromadb
import numpy as np

c = chromadb.PersistentClient(path=r'D:\project\rag-project\chroma_experiments')
col = c.get_collection('v2_sentence_300')

result = col.get(where={'source': '2601.03085v1.pdf'}, limit=3, include=['embeddings', 'documents'])

for i, (emb, doc) in enumerate(zip(result['embeddings'], result['documents'])):
    emb = np.array(emb)
    print(f'--- chunk {i} ---')
    print('text snippet:', doc[:150])
    print('embedding norm:', np.linalg.norm(emb))
    print('first 5 values:', emb[:5])
    print()