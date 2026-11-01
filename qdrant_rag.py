# qdrant_rag.py
import os
import re  # <--- ДОБАВЛЕН ИМПОРТ
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger("RAG_Qdrant")

class QdrantRAGEngine:
    def __init__(self, txt_path: str, db_path: str = "qdrant_db"):
        self.txt_path = txt_path
        self.collection_name = "knowledge_base"
        
        logger.info("Загрузка модели эмбеддингов...")
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-small')
        
        self.client = QdrantClient(path=db_path)
        
        self.chunks = []
        self.bm25 = None
        
        self._init_collection()
        self._index_data()
        self._init_bm25()

    def _init_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

    def _load_chunks(self):
        if not os.path.exists(self.txt_path):
            logger.error(f"Файл {self.txt_path} не найден!")
            return []
        with open(self.txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Разбиваем по предложениям
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(sentence) > 1000:
                sentence = sentence[:1000]
                
            if len(current_chunk) + len(sentence) < 400:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) > 50:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk and len(current_chunk) > 50:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _index_data(self):
        self.chunks = self._load_chunks()
        count = self.client.count(collection_name=self.collection_name).count
        
        if count == 0 and self.chunks:
            logger.info(f"Индексация {len(self.chunks)} кусков в Qdrant...")
            points = []
            for i, chunk in enumerate(self.chunks):
                vector = self.embedder.encode(chunk).tolist()
                points.append(
                    models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk})
                )
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("Индексация Qdrant завершена!")
        else:
            logger.info(f"База Qdrant уже содержит {count} векторов.")

    def _init_bm25(self):
        """Инициализация лексического поиска (BM25)"""
        if not self.chunks:
            return
        logger.info("Инициализация BM25 (точный поиск по словам)...")
        # УЛУЧШЕНИЕ: Извлекаем только буквы и цифры (без знаков препинания)
        tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_context(self, question: str) -> str:
        if not self.chunks:
            return "База знаний пуста."

        # 1. Векторный поиск (Qdrant)
        q_emb = self.embedder.encode(question).tolist()
        qdrant_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=q_emb, 
            limit=10
        ).points
        qdrant_texts = [hit.payload["text"] for hit in qdrant_hits]

        # 2. Лексический поиск (BM25)
        # УЛУЧШЕНИЕ: Очищаем вопрос от знаков препинания
        tokenized_query = re.findall(r'\w+', question.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        bm25_texts = [self.chunks[i] for i in top_bm25_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60

        for rank, text in enumerate(qdrant_texts):
            rrf_scores[text] = rrf_scores.get(text, 0) + (1 / (k + rank + 1))

        for rank, text in enumerate(bm25_texts):
            rrf_scores[text] = rrf_scores.get(text, 0) + (1 / (k + rank + 1))

        # Берем топ-3
        best_texts = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        
        return "\n\n".join([text for text, score in best_texts])