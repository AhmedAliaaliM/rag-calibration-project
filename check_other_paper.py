import chromadb

c = chromadb.PersistentClient(path=r'D:\project\rag-project\chroma_experiments')
col = c.get_collection('v2_sentence_300')

result = col.get(where={'source': '2510.19012v1.pdf'}, limit=3, include=['documents'])
for i, doc in enumerate(result['documents']):
    print(f'--- chunk {i} ---')
    print(doc[:150])
    print()