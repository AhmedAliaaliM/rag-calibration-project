import chromadb

c = chromadb.PersistentClient(path=r'D:\project\rag-project\chroma_experiments')
col = c.get_collection('v2_sentence_300')

before = col.count()
print(f'Chunks before deletion: {before}')

col.delete(where={'source': '2601.03085v1.pdf'})

after = col.count()
print(f'Chunks after deletion: {after}')
print(f'Removed: {before - after} chunks')