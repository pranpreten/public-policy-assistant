"""
RAG 파이프라인 모듈
- 질문 분류 (정책 vs 일반)
- 정책 질문 → RAG 기반 답변
- 일반 질문 → LLM 직접 답변
- 출처 표시 기능 포함
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.llm_model import PolicyLLM


# 정책 관련 키워드
POLICY_KEYWORDS = [
    "미세먼지", "폭염", "비상저감", "대기오염", "경보", "주의보",
    "법령", "법률", "조례", "지침", "규정", "규칙", "조항", "제조",
    "환경부", "고용노동부", "교육부", "국토교통부", "행정안전부",
    "산업안전", "근로자", "사업장", "건설현장", "공사장",
    "학교", "휴업", "수업", "실외활동",
    "온열질환", "열사병", "폭염특보",
    "PM2.5", "PM10", "초미세먼지",
    "발령", "해제", "기준", "조치", "대응", "매뉴얼",
    "취약계층", "어린이", "노인", "임산부"
]


@dataclass
class RAGResponse:
    """RAG 응답 데이터 클래스"""
    answer: str
    sources: List[Dict[str, Any]]
    question: str
    is_policy_question: bool = True


class PolicyRAG:
    """공공 정책 RAG 시스템 (일반 질문도 지원)"""

    def __init__(
        self,
        vectordb_path: str = "./vectordb",
        llm_model: str = "Qwen/Qwen2.5-14B-Instruct",
        llm_model_path: str = None,  # 로컬 모델 경로
        embedding_model: str = "jhgan/ko-sroberta-multitask"
    ):
        """
        Args:
            vectordb_path: 벡터 DB 경로
            llm_model: LLM 모델 이름 (HuggingFace)
            llm_model_path: 로컬 모델 경로 (있으면 우선 사용)
            embedding_model: 임베딩 모델
        """
        print("=" * 60)
        print("공공 정책 RAG 시스템 초기화 (vLLM)")
        print("=" * 60)

        self.similarity_threshold = 0.7  # 유사도 임계값

        # 벡터 스토어 초기화
        print("\n[1/2] 벡터 스토어 초기화...")
        self.vector_store = VectorStore(
            persist_directory=vectordb_path,
            embedding_model=embedding_model
        )
        stats = self.vector_store.get_collection_stats()
        print(f"  - 로드된 문서 수: {stats['total_documents']}")

        # LLM 초기화 (vLLM)
        print("\n[2/2] vLLM 모델 초기화...")
        self.llm = PolicyLLM(
            model_name=llm_model,
            model_path=llm_model_path
        )

        print("\n" + "=" * 60)
        print("RAG 시스템 초기화 완료!")
        print("=" * 60)

    def is_policy_question(self, question: str) -> bool:
        """질문이 정책 관련인지 판단"""
        question_lower = question.lower()

        # 키워드 기반 판단
        for keyword in POLICY_KEYWORDS:
            if keyword in question_lower or keyword.lower() in question_lower:
                return True

        return False

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_dict=filter_dict
        )
        return self.vector_store.format_search_results(results)

    def generate_policy_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        max_new_tokens: int = 1024
    ) -> str:
        """정책 질문: RAG 기반 답변 생성"""
        prompt = self.llm.create_qa_prompt(question, context_docs)
        return self.llm.generate_response(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7
        )

    def generate_general_answer(
        self,
        question: str,
        max_new_tokens: int = 1024
    ) -> str:
        """일반 질문: LLM 직접 답변"""
        prompt = f"""<|im_start|>system
당신은 친절하고 유능한 AI 어시스턴트입니다. 사용자의 질문에 정확하고 도움이 되는 답변을 제공합니다.
한국어로 자연스럽게 대화하세요.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        return self.llm.generate_response(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7
        )

    def answer(
        self,
        question: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> RAGResponse:
        """
        질문에 대한 답변 생성
        - 정책 질문: RAG 파이프라인
        - 일반 질문: LLM 직접 응답
        """
        print(f"\n🔍 질문: {question}")

        # 1. 질문 분류
        is_policy = self.is_policy_question(question)

        if is_policy:
            print("📋 정책 관련 질문 감지 → RAG 모드")
            print("📚 관련 문서 검색 중...")
            docs = self.retrieve(question, n_results, filter_dict)

            if docs:
                print(f"  - {len(docs)}개 관련 문서 발견")
                print("💬 답변 생성 중...")
                answer = self.generate_policy_answer(question, docs)

                # 출처 정보 정리
                sources = []
                seen = set()
                for doc in docs:
                    meta = doc['metadata']
                    source_key = f"{meta['source']}-{meta['doc_title']}"
                    if source_key not in seen:
                        seen.add(source_key)
                        sources.append({
                            'source': meta['source'],
                            'doc_title': meta['doc_title'],
                            'doc_type': meta['doc_type'],
                            'law_reference': meta['law_reference'],
                            'effective_date': meta['effective_date']
                        })

                return RAGResponse(
                    answer=answer,
                    sources=sources,
                    question=question,
                    is_policy_question=True
                )

        # 일반 질문 또는 관련 문서 없음
        print("💬 일반 질문 → LLM 직접 응답")
        answer = self.generate_general_answer(question)

        return RAGResponse(
            answer=answer,
            sources=[],
            question=question,
            is_policy_question=False
        )

    def format_response(self, response: RAGResponse) -> str:
        """응답을 보기 좋게 포맷팅"""
        output = []
        output.append("=" * 60)
        output.append(f"❓ 질문: {response.question}")
        output.append("=" * 60)
        output.append("")
        output.append("💡 답변:")
        output.append("-" * 40)
        output.append(response.answer)

        if response.sources:
            output.append("")
            output.append("-" * 40)
            output.append("📋 참고 출처:")
            for i, src in enumerate(response.sources, 1):
                output.append(f"  [{i}] {src['source']} - {src['doc_title']}")
                output.append(f"      관련 법령: {src['law_reference']}")
                output.append(f"      시행일: {src['effective_date']}")
        else:
            output.append("")
            output.append("ℹ️ 일반 질문 - LLM 직접 응답")

        output.append("=" * 60)
        return "\n".join(output)


def main():
    """RAG 시스템 테스트"""
    base_dir = Path(__file__).parent.parent
    vectordb_path = str(base_dir / "vectordb")

    # RAG 시스템 초기화 (vLLM)
    rag = PolicyRAG(
        vectordb_path=vectordb_path,
        llm_model="Qwen/Qwen2.5-14B-Instruct"
    )

    # 테스트 질문들 (정책 + 일반)
    test_questions = [
        "미세먼지 비상저감조치 발령 기준이 뭐야?",
        "기온 35도 이상이면 산업 현장 어떤 조치 필요해?",
        "오늘 날씨 어때?",  # 일반 질문
        "파이썬이 뭐야?",  # 일반 질문
    ]

    print("\n" + "=" * 60)
    print("테스트 질의응답 시작")
    print("=" * 60)

    for question in test_questions:
        response = rag.answer(question)
        print(rag.format_response(response))
        print("\n")


if __name__ == "__main__":
    main()
