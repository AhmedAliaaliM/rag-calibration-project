import chromadb

c = chromadb.PersistentClient(path=r'D:\project\rag-project\chroma_experiments')
col = c.get_collection('v2_sentence_300')
col.delete(ids=["poisoned_test_chunk_1"])
print("Poisoned chunk removed.")