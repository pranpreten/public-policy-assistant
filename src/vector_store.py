"""
벡터 DB 모듈
- ChromaDB를 사용한 벡터 저장소
- 한국어 임베딩 모델 사용
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """벡터 저장소 클래스"""

    def __init__(
        self,
        persist_directory: str = "./vectordb",
        embedding_model: str = "jhgan/ko-sroberta-multitask"  # 한국어 전용 SBERT
    ):
        """
        Args:
            persist_directory: ChromaDB 저장 경로
            embedding_model: 사용할 임베딩 모델 (한국어 지원 모델)
        """
        self.persist_directory = persist_directory

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=persist_directory)

        # 임베딩 모델 로드
        print(f"임베딩 모델 로딩: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("임베딩 모델 로딩 완료")

        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name="policy_documents",
            metadata={"description": "공공 정책 및 법령 문서"}
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩 벡터로 변환"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """문서 청크들을 벡터 DB에 추가"""
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk['chunk_id'])
            documents.append(chunk['content'])
            metadatas.append({
                'source': chunk['source'],
                'doc_title': chunk['doc_title'],
                'doc_type': chunk['doc_type'],
                'section': chunk['section'],
                'law_reference': chunk['law_reference'],
                'effective_date': chunk['effective_date']
            })

        # 임베딩 생성
        print(f"총 {len(documents)}개 문서 임베딩 중...")
        embeddings = self.embed_texts(documents)

        # ChromaDB에 추가
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"벡터 DB에 {len(documents)}개 문서 추가 완료")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """유사 문서 검색"""
        # 쿼리 임베딩
        query_embedding = self.embed_texts([query])[0]

        # 검색 실행
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )

        return results

    def format_search_results(self, results: Dict[str, Any]) -> List[Dict]:
        """검색 결과를 포맷팅"""
        formatted = []

        if not results['documents'] or not results['documents'][0]:
            return formatted

        for i, doc in enumerate(results['documents'][0]):
            formatted.append({
                'content': doc,
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'distance': results['distances'][0][i] if results.get('distances') else None
            })

        return formatted

    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 반환"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }


def build_vector_store():
    """벡터 스토어 구축"""
    base_dir = Path(__file__).parent.parent
    chunks_path = base_dir / "data" / "processed" / "chunks.json"
    vectordb_path = str(base_dir / "vectordb")

    # 청크 데이터 로드
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"총 {len(chunks)}개 청크 로드됨")

    # 벡터 스토어 초기화 및 문서 추가
    store = VectorStore(persist_directory=vectordb_path)

    # 기존 데이터가 있으면 스킵
    stats = store.get_collection_stats()
    if stats['total_documents'] > 0:
        print(f"기존 벡터 DB 발견: {stats['total_documents']}개 문서")
        print("기존 데이터를 삭제하고 새로 구축합니다...")
        store.client.delete_collection("policy_documents")
        store.collection = store.client.get_or_create_collection(
            name="policy_documents",
            metadata={"description": "공공 정책 및 법령 문서"}
        )

    store.add_documents(chunks)

    # 테스트 검색
    print("\n=== 테스트 검색 ===")
    test_queries = [
        "미세먼지 비상저감조치 발령 기준이 뭐야?",
        "기온 35도 이상이면 산업 현장 어떤 조치 필요해?",
        "학교에서 미세먼지 나쁨이면 어떻게 해야 해?"
    ]

    for query in test_queries:
        print(f"\n질문: {query}")
        results = store.search(query, n_results=2)
        formatted = store.format_search_results(results)
        for i, result in enumerate(formatted):
            print(f"\n  [{i+1}] 출처: {result['metadata']['source']} - {result['metadata']['doc_title']}")
            print(f"      조항: {result['metadata']['section']}")
            print(f"      내용: {result['content'][:100]}...")


if __name__ == "__main__":
    build_vector_store()
