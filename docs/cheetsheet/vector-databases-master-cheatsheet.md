---
title: Vector Databases Master Cheatsheet
sidebar_position: 11
---

# Vector Databases Master Cheatsheet

Vector databases store embeddings and support fast similarity search — the backbone of RAG, semantic search, recommendation, and clustering. Covers FAISS (local, in-process), Chroma (local persistent), Pinecone, Weaviate, Qdrant, and pgvector.

## Concepts and distance metrics

| Concept | Description | Code/example |
|---|---|---|
| Embedding | A dense float vector representing a piece of content. Produced by an encoder (Sentence-BERT, OpenAI, Cohere). Typical dim: 384, 768, 1024, 1536, 3072. | `vec = embedder.encode("Hello world")  # shape: (768,)` |
| Cosine similarity | Score: dot product divided by the product of vector norms. Default for embeddings; normalize to unit length and it equals dot product. | `def cosine(a, b):`<br/>`    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))` |
| Euclidean (L2) distance | $\sqrt{\sum_i (a_i - b_i)^2}$ — Smaller is more similar. Equivalent to cosine for unit-normalized vectors. | `dist = np.linalg.norm(a - b)` |
| Inner product (dot) | $a \cdot b$ — Faster than cosine. Use when vectors are normalized OR when magnitude carries meaning. | `score = a @ b` |
| ANN (Approximate Nearest Neighbor) | Trade a small recall loss for huge speedups. Algorithms: HNSW, IVF, PQ, ScaNN. | Configurable in every vector DB — never search >100K vectors exhaustively in production. |
| Index types | **Flat** (exact, slow for >100K), **IVF** (cluster-then-search), **HNSW** (graph-based, default for most), **PQ/IVFPQ** (compressed, memory-saving). | `# FAISS: IndexFlatIP, IndexIVFFlat, IndexHNSWFlat, IndexIVFPQ` |
| Hybrid search | Combine vector (semantic) + BM25 (keyword) search. Reciprocal Rank Fusion (RRF) is the standard combiner. | Most modern DBs (Weaviate, Qdrant, pgvector + tsvector) support both natively. |
| Reranking | Use a cross-encoder to rescore top-k retrieved results before returning. Significantly improves quality. | `from sentence_transformers import CrossEncoder`<br/>`reranker = CrossEncoder("BAAI/bge-reranker-base")` |

## FAISS (local, in-process)

| Method | Description | Code example |
|---|---|---|
| Install | `pip install faiss-cpu` or `pip install faiss-gpu` for CUDA. | `pip install faiss-cpu` |
| `IndexFlatIP` | `faiss.IndexFlatIP(d)` — Exact inner-product search. Use for normalized vectors → cosine equivalent. | `import faiss, numpy as np`<br/>`index = faiss.IndexFlatIP(768)`<br/>`index.add(vectors.astype("float32"))` |
| `IndexFlatL2` | `faiss.IndexFlatL2(d)` — Exact L2 search. | `index = faiss.IndexFlatL2(768)` |
| `IndexHNSWFlat` | `faiss.IndexHNSWFlat(d, M=32)` — Approximate via HNSW graph. `M` = max neighbors per node. | `index = faiss.IndexHNSWFlat(768, 32)`<br/>`index.hnsw.efConstruction = 200`<br/>`index.hnsw.efSearch = 64` |
| `IndexIVFFlat` | `faiss.IndexIVFFlat(quantizer, d, nlist)` — Inverted file. Needs `.train(...)` first. | `quantizer = faiss.IndexFlatL2(768)`<br/>`index = faiss.IndexIVFFlat(quantizer, 768, nlist=100)`<br/>`index.train(vectors)`<br/>`index.add(vectors)` |
| `IndexIVFPQ` | `faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)` — Product Quantization. Massive memory savings. | `index = faiss.IndexIVFPQ(quantizer, 768, 100, M=8, nbits=8)`<br/>`index.train(vectors)` |
| `.add()` | `index.add(vectors)` — Add float32 vectors to the index. Shape `(N, d)`. | `index.add(np.asarray(vecs, dtype=np.float32))` |
| `.search()` | `index.search(query_vectors, k)` — Returns `(distances, indices)`, both shape `(num_queries, k)`. | `D, I = index.search(query.reshape(1, -1).astype("float32"), k=5)`<br/>`print(I[0], D[0])` |
| `.remove_ids()` | `index.remove_ids(faiss.IDSelectorBatch(np.array([id1, id2])))` — Delete vectors by ID. | `index.remove_ids(faiss.IDSelectorBatch(np.array([42, 99])))` |
| Save / load | `faiss.write_index(index, path)` / `faiss.read_index(path)`. | `faiss.write_index(index, "my.index")`<br/>`index = faiss.read_index("my.index")` |
| GPU acceleration | `faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)` — Move index to GPU 0. | `res = faiss.StandardGpuResources()`<br/>`gpu_index = faiss.index_cpu_to_gpu(res, 0, index)` |
| Normalize vectors | `faiss.normalize_L2(vectors)` — In-place L2 normalization. Combine with `IndexFlatIP` for cosine. | `faiss.normalize_L2(vectors)`<br/>`index.add(vectors)` |

## Chroma (local persistent)

| Method | Description | Code example |
|---|---|---|
| Install | `pip install chromadb` — Embedded mode runs in-process; client/server mode for production. | `pip install chromadb` |
| `chromadb.PersistentClient()` | `chromadb.PersistentClient(path='./chroma_db', settings=None)` — Disk-backed local instance. | `import chromadb`<br/>`client = chromadb.PersistentClient(path="./chroma")` |
| `chromadb.Client()` | In-memory client (volatile). For tests / notebooks. | `client = chromadb.Client()` |
| HTTP client | `chromadb.HttpClient(host='localhost', port=8000, ssl=False, headers=None)` — Connect to a Chroma server. | `client = chromadb.HttpClient(host="localhost", port=8000)` |
| `.create_collection()` | `client.create_collection(name, embedding_function=None, metadata=None)` — Make a new collection. | `coll = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})` |
| `.get_or_create_collection()` | Convenience: get if exists, else create. | `coll = client.get_or_create_collection("docs")` |
| `.add()` | `coll.add(documents=None, embeddings=None, metadatas=None, ids=None)` — Insert. Provide IDs explicitly. | `coll.add(`<br/>`    documents=["doc 1", "doc 2"],`<br/>`    embeddings=[vec1, vec2],`<br/>`    metadatas=[{"src": "a"}, {"src": "b"}],`<br/>`    ids=["1", "2"],`<br/>`)` |
| `.upsert()` | Same args as `.add()` — Adds new or overwrites existing IDs. | `coll.upsert(ids=["1"], embeddings=[new_vec], documents=["updated"])` |
| `.query()` | `coll.query(query_embeddings=None, query_texts=None, n_results=10, where=None, where_document=None, include=['documents', 'metadatas', 'distances'])`. | `results = coll.query(`<br/>`    query_embeddings=[q_vec],`<br/>`    n_results=5,`<br/>`    where={"src": "a"},`<br/>`)` |
| Metadata filter | `where={"key": value}` or operators: `{"price": {"$gt": 100}}`, `{"$and": [...]}`, `{"$or": [...]}`. | `coll.query(query_embeddings=[v], where={"$and": [{"year": {"$gte": 2024}}, {"lang": "en"}]})` |
| Full-text filter | `where_document={"$contains": "term"}` — Search within document text. | `coll.query(query_embeddings=[v], where_document={"$contains": "neural network"})` |
| `.get()` | `coll.get(ids=None, where=None, limit=None, offset=None, include=[...])` — Fetch by ID or filter (no vector search). | `coll.get(ids=["1", "2"])` |
| `.update()` | `coll.update(ids, embeddings=None, metadatas=None, documents=None)` — Modify existing entries. | `coll.update(ids=["1"], metadatas=[{"src": "updated"}])` |
| `.delete()` | `coll.delete(ids=None, where=None)` — Remove by ID or filter. | `coll.delete(ids=["1", "2"])` |
| `.count()` | `coll.count() -> int` — Number of vectors. | `print(coll.count())` |
| Built-in embedding fn | Pass `embedding_function=OpenAIEmbeddingFunction(api_key=...)`, `SentenceTransformerEmbeddingFunction(model_name=...)`, etc. | `from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction`<br/>`ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")`<br/>`coll = client.create_collection("docs", embedding_function=ef)`<br/>`coll.add(documents=["text"], ids=["1"])  # embeds automatically` |

## Pinecone (managed cloud)

| Method | Description | Code example |
|---|---|---|
| Install | `pip install pinecone` (modern SDK) — Replaces older `pinecone-client`. | `pip install pinecone` |
| Init | `Pinecone(api_key=None)` — Reads `PINECONE_API_KEY` env var if unset. | `from pinecone import Pinecone`<br/>`pc = Pinecone(api_key="...")` |
| Create index | `pc.create_index(name, dimension, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))`. | `from pinecone import ServerlessSpec`<br/>`pc.create_index(`<br/>`    name="my-index",`<br/>`    dimension=1536,`<br/>`    metric="cosine",`<br/>`    spec=ServerlessSpec(cloud="aws", region="us-east-1"),`<br/>`)` |
| Connect to index | `index = pc.Index(name)` — Get an `Index` object for the operations below. | `index = pc.Index("my-index")` |
| `.upsert()` | `index.upsert(vectors, namespace='', batch_size=100, async_req=False)` — Insert or update. Vectors as `(id, values, metadata)` tuples or dicts. | `index.upsert(vectors=[`<br/>`    {"id": "1", "values": vec1, "metadata": {"src": "doc1"}},`<br/>`    {"id": "2", "values": vec2, "metadata": {"src": "doc2"}},`<br/>`])` |
| `.query()` | `index.query(vector=None, id=None, top_k=10, namespace='', filter=None, include_values=False, include_metadata=False)`. | `result = index.query(`<br/>`    vector=q_vec,`<br/>`    top_k=5,`<br/>`    include_metadata=True,`<br/>`    filter={"src": {"$eq": "doc1"}},`<br/>`)` |
| Metadata filters | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`. | `filter={"year": {"$gte": 2024}, "lang": {"$in": ["en", "fr"]}}` |
| `.fetch()` | `index.fetch(ids, namespace='')` — Get by IDs without similarity search. | `index.fetch(ids=["1", "2"])` |
| `.update()` | `index.update(id, values=None, set_metadata=None, namespace='')` — Modify a single vector or its metadata. | `index.update(id="1", set_metadata={"src": "updated"})` |
| `.delete()` | `index.delete(ids=None, namespace='', filter=None, delete_all=False)` — Delete by IDs or filter. | `index.delete(ids=["1", "2"])`<br/>`index.delete(filter={"src": {"$eq": "stale"}})` |
| Namespaces | `namespace=` argument on every operation — Logical partitions inside one index. Common for multi-tenant. | `index.upsert(vectors=[...], namespace="user-42")` |
| `.describe_index_stats()` | `index.describe_index_stats(filter=None) -> dict` — Total count, namespaces, dimensions. | `print(index.describe_index_stats())` |
| Delete index | `pc.delete_index(name)`. | `pc.delete_index("my-index")` |

## Qdrant

| Method | Description | Code example |
|---|---|---|
| Install | `pip install qdrant-client` — Both local (`:memory:` or path) and remote (URL + API key). | `pip install qdrant-client` |
| Client | `QdrantClient(url=None, api_key=None, path=None)` — Use `path=":memory:"` for in-memory, `path=./qdrant_data` for local persistent. | `from qdrant_client import QdrantClient`<br/>`client = QdrantClient(":memory:")` |
| Cloud client | `QdrantClient(url='https://...', api_key='...')`. | `client = QdrantClient(url="https://xyz.qdrant.io", api_key="...")` |
| Create collection | `client.create_collection(collection_name, vectors_config=VectorParams(size=, distance=))`. | `from qdrant_client.models import VectorParams, Distance`<br/>`client.create_collection(`<br/>`    "docs",`<br/>`    vectors_config=VectorParams(size=768, distance=Distance.COSINE),`<br/>`)` |
| `.upsert()` | `client.upsert(collection_name, points, wait=True)` where `points = [PointStruct(id, vector, payload)]`. | `from qdrant_client.models import PointStruct`<br/>`client.upsert(`<br/>`    "docs",`<br/>`    points=[`<br/>`        PointStruct(id=1, vector=vec1, payload={"src": "a"}),`<br/>`        PointStruct(id=2, vector=vec2, payload={"src": "b"}),`<br/>`    ],`<br/>`)` |
| `.search()` | `client.search(collection_name, query_vector, limit=10, query_filter=None, with_payload=True, with_vectors=False)`. | `results = client.search(`<br/>`    "docs",`<br/>`    query_vector=q_vec,`<br/>`    limit=5,`<br/>`    with_payload=True,`<br/>`)` |
| Payload filter | `Filter(must=[FieldCondition(key='src', match=MatchValue(value='a'))])`. | `from qdrant_client.models import Filter, FieldCondition, MatchValue`<br/>`flt = Filter(must=[FieldCondition(key="src", match=MatchValue(value="a"))])`<br/>`client.search("docs", query_vector=q_vec, query_filter=flt)` |
| Range filter | `Range(gte=2024, lte=2026)` on a numeric field. | `from qdrant_client.models import Range`<br/>`Filter(must=[FieldCondition(key="year", range=Range(gte=2024))])` |
| `.delete()` | `client.delete(collection_name, points_selector=...)` — Selector by IDs or filter. | `from qdrant_client.models import PointIdsList`<br/>`client.delete("docs", points_selector=PointIdsList(points=[1, 2]))` |
| `.retrieve()` | `client.retrieve(collection_name, ids, with_payload=True, with_vectors=False)` — Get by IDs. | `client.retrieve("docs", ids=[1, 2])` |
| `.count()` | `client.count(collection_name, count_filter=None, exact=True)`. | `print(client.count("docs"))` |

## Weaviate

| Method | Description | Code example |
|---|---|---|
| Install | `pip install weaviate-client` (v4 API). | `pip install weaviate-client` |
| Connect | `weaviate.connect_to_local(host='localhost', port=8080)` / `weaviate.connect_to_weaviate_cloud(cluster_url=..., auth_credentials=...)`. | `import weaviate`<br/>`client = weaviate.connect_to_local()` |
| Create collection | `client.collections.create(name, properties=[...], vectorizer_config=..., vector_index_config=...)`. | `from weaviate.classes.config import Property, DataType, Configure`<br/>`client.collections.create(`<br/>`    name="Articles",`<br/>`    properties=[Property(name="title", data_type=DataType.TEXT)],`<br/>`    vectorizer_config=Configure.Vectorizer.text2vec_openai(),`<br/>`)` |
| Get collection | `coll = client.collections.get(name)` — Returns the collection handle. | `coll = client.collections.get("Articles")` |
| Insert | `coll.data.insert(properties={...}, uuid=None, vector=None)` — Single insert. | `coll.data.insert(properties={"title": "Vector DBs"})` |
| Batch insert | `with coll.batch.dynamic() as batch: batch.add_object(properties={...}, uuid=..., vector=...)`. | `with coll.batch.dynamic() as batch:`<br/>`    for doc in docs:`<br/>`        batch.add_object(properties={"title": doc.title}, vector=doc.vector)` |
| Vector search | `coll.query.near_vector(near_vector=..., limit=10, filters=None, return_metadata=MetadataQuery(distance=True))`. | `from weaviate.classes.query import MetadataQuery`<br/>`results = coll.query.near_vector(`<br/>`    near_vector=q_vec,`<br/>`    limit=5,`<br/>`    return_metadata=MetadataQuery(distance=True),`<br/>`)` |
| Hybrid search (vector + BM25) | `coll.query.hybrid(query=..., vector=None, alpha=0.5, limit=10)` — `alpha=0` is pure BM25, `alpha=1` is pure vector. | `coll.query.hybrid(query="neural network", alpha=0.7, limit=5)` |
| BM25 only | `coll.query.bm25(query=..., limit=10, query_properties=['title'])`. | `coll.query.bm25(query="attention", limit=5)` |
| Filter | `Filter.by_property('field').equal(value)` / `.greater_than(x)` / `.contains_any([...])`. | `from weaviate.classes.query import Filter`<br/>`coll.query.near_vector(near_vector=q_vec, filters=Filter.by_property("year").greater_than(2024))` |
| Close connection | Always call `client.close()` at end (v4 holds a gRPC connection). | `client.close()` |

## pgvector (Postgres extension)

| Method | Description | Code example |
|---|---|---|
| Install | `CREATE EXTENSION vector;` on the Postgres database. Python: `pip install psycopg pgvector`. | `CREATE EXTENSION IF NOT EXISTS vector;` |
| Column type | `vector(d)` — Add a vector column of dimension `d`. | `CREATE TABLE items (`<br/>`    id BIGSERIAL PRIMARY KEY,`<br/>`    content TEXT,`<br/>`    embedding vector(768)`<br/>`);` |
| Distance operators | `<->` (L2), `<=>` (cosine), `<#>` (negative inner product). | `SELECT * FROM items ORDER BY embedding <-> '[1,2,3,...]' LIMIT 5;` |
| Index (HNSW) | `CREATE INDEX ... USING hnsw (col vector_<distance>_ops)` — Use `vector_cosine_ops` for cosine, `vector_l2_ops` for L2, `vector_ip_ops` for inner product. | `CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);` |
| Index (IVFFlat) | `CREATE INDEX ... USING ivfflat (col vector_cosine_ops) WITH (lists=100)` — Cheaper than HNSW; tune `lists`. | `CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);` |
| Filter + similarity | Combine `WHERE` with ORDER BY embedding distance. | `SELECT id, content FROM items WHERE year > 2024 ORDER BY embedding <=> :q LIMIT 10;` |
| Insert | Standard SQL — pass vector as Python list (with psycopg/`pgvector` adapter registered). | `cur.execute("INSERT INTO items (content, embedding) VALUES (%s, %s)", (text, vec))` |
| Update | `UPDATE items SET embedding = %s WHERE id = %s;`. | `cur.execute("UPDATE items SET embedding = %s WHERE id = %s", (new_vec, 42))` |
| Halfvec (Postgres 0.7+) | `halfvec(d)` — Half-precision float (16-bit), 2× smaller memory. | `ALTER TABLE items ADD COLUMN embedding_half halfvec(768);` |
| Sparse vectors | `sparsevec(d)` — For very high-dim sparse representations (BM25-like, SPLADE). | `embedding sparsevec(30000)` |

## Reranking

| Method | Description | Code example |
|---|---|---|
| Cross-encoder reranker | `CrossEncoder(model_name)` from `sentence-transformers` — Takes (query, doc) pairs, returns relevance scores. Use `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`. | `from sentence_transformers import CrossEncoder`<br/>`reranker = CrossEncoder("BAAI/bge-reranker-base")`<br/>`pairs = [(query, doc.text) for doc in candidates]`<br/>`scores = reranker.predict(pairs)`<br/>`reranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]` |
| Cohere rerank API | `cohere.Client(api_key).rerank(query, documents, top_n=5, model='rerank-english-v3.0')`. | `import cohere`<br/>`co = cohere.Client("...")`<br/>`result = co.rerank(model="rerank-english-v3.0", query=q, documents=docs, top_n=5)` |
| Reciprocal Rank Fusion (RRF) | Combine multiple rankings: $score(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}$ where $k \approx 60$. | `def rrf(rankings, k=60):`<br/>`    scores = {}`<br/>`    for ranking in rankings:`<br/>`        for rank, doc_id in enumerate(ranking, 1):`<br/>`            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)`<br/>`    return sorted(scores.items(), key=lambda x: -x[1])` |

## Common patterns

| Pattern | Code |
|---|---|
| End-to-end FAISS RAG (no DB) | `import faiss, numpy as np`<br/>`from sentence_transformers import SentenceTransformer`<br/>`embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")`<br/>`vecs = embedder.encode(docs, normalize_embeddings=True).astype("float32")`<br/>`index = faiss.IndexFlatIP(vecs.shape[1])`<br/>`index.add(vecs)`<br/>`q = embedder.encode("query", normalize_embeddings=True).astype("float32").reshape(1, -1)`<br/>`D, I = index.search(q, k=5)`<br/>`top_docs = [docs[i] for i in I[0]]` |
| Persistent Chroma + auto-embedding | `import chromadb`<br/>`from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction`<br/>`client = chromadb.PersistentClient(path="./chroma")`<br/>`ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")`<br/>`coll = client.get_or_create_collection("docs", embedding_function=ef)`<br/>`coll.add(documents=docs, ids=[str(i) for i in range(len(docs))])`<br/>`result = coll.query(query_texts=["query"], n_results=5)` |
| Pinecone with metadata filter | `pc = Pinecone()`<br/>`index = pc.Index("my-index")`<br/>`index.upsert(vectors=[{"id": "1", "values": vec, "metadata": {"year": 2025, "lang": "en"}}])`<br/>`r = index.query(vector=q, top_k=5, filter={"year": {"$gte": 2024}, "lang": "en"}, include_metadata=True)` |
| Qdrant hybrid setup | `client.create_collection("docs", vectors_config=VectorParams(size=768, distance=Distance.COSINE))`<br/>`# upsert with payload, then search with filter`<br/>`results = client.search("docs", query_vector=q, query_filter=Filter(must=[FieldCondition(key="src", match=MatchValue(value="a"))]), limit=10)` |
| pgvector with HNSW index | `CREATE EXTENSION vector;`<br/>`CREATE TABLE docs (id BIGSERIAL PRIMARY KEY, content TEXT, embedding vector(768));`<br/>`CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);`<br/>`# Insert via psycopg + pgvector adapter`<br/>`SELECT id FROM docs ORDER BY embedding <=> :q LIMIT 5;` |
| Retrieve + rerank pipeline | `candidates = vector_store.similarity_search(query, k=50)  # over-fetch`<br/>`pairs = [(query, c.page_content) for c in candidates]`<br/>`scores = reranker.predict(pairs)`<br/>`top_k = [c for _, c in sorted(zip(scores, candidates), reverse=True)[:5]]` |
| Hybrid search via RRF | `vector_hits = vector_store.search(q, k=20)`<br/>`bm25_hits = bm25.search(q, k=20)`<br/>`fused = rrf([[h.id for h in vector_hits], [h.id for h in bm25_hits]])`<br/>`top = [doc_by_id[doc_id] for doc_id, _ in fused[:5]]` |
| Update embedding pipeline (sync new docs) | `existing_ids = set(coll.get(include=[])["ids"])`<br/>`new_docs = [d for d in all_docs if d.id not in existing_ids]`<br/>`if new_docs:`<br/>`    coll.add(documents=[d.text for d in new_docs], ids=[d.id for d in new_docs])` |
| Bulk delete by metadata | `# Chroma`<br/>`coll.delete(where={"src": "stale-source"})`<br/>`# Pinecone`<br/>`index.delete(filter={"src": {"$eq": "stale-source"}})` |
| Choosing dimension and metric | `# Match the embedder. OpenAI text-embedding-3-small: 1536. bge-small: 384.`<br/>`# Cosine is the default. Use inner product (IP) only with normalized vectors.`<br/>`# Larger dim → more accurate but slower + more memory.` |
| Backup / migrate | Most DBs support snapshot/export. For Chroma: copy the persist directory. For Pinecone: use `.fetch()` in batches. For pgvector: standard `pg_dump`. | |
