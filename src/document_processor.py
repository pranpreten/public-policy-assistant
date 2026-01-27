"""
문서 처리 및 청킹 모듈
- 문서를 의미 단위로 분할
- 메타데이터 유지
- 출처 정보 보존
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class DocumentChunk:
    """문서 청크 데이터 클래스"""
    chunk_id: str
    content: str
    source: str  # 출처 기관
    doc_title: str  # 문서 제목
    doc_type: str  # 법령/지침
    section: str  # 조항/섹션
    law_reference: str  # 관련 법령 조문
    effective_date: str  # 시행일


class DocumentProcessor:
    """문서 처리 및 청킹 클래스"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_policy_data(self, data_path: Path) -> Dict:
        """정책 데이터 로드"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_sections(self, content: str) -> List[Dict[str, str]]:
        """마크다운 형식의 콘텐츠에서 섹션 추출"""
        sections = []

        # ## 또는 ### 헤딩 기준으로 분리
        pattern = r'(#{2,4}\s+[^\n]+)'
        parts = re.split(pattern, content)

        current_section = ""
        current_content = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(r'^#{2,4}\s+', part):
                # 이전 섹션 저장
                if current_section and current_content.strip():
                    sections.append({
                        "section": current_section,
                        "content": current_content.strip()
                    })
                current_section = part.lstrip('#').strip()
                current_content = ""
            else:
                current_content += part + "\n"

        # 마지막 섹션 저장
        if current_section and current_content.strip():
            sections.append({
                "section": current_section,
                "content": current_content.strip()
            })

        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections and content.strip():
            sections.append({
                "section": "전체",
                "content": content.strip()
            })

        return sections

    def chunk_text(self, text: str, max_length: int = None) -> List[str]:
        """텍스트를 적절한 크기로 분할"""
        if max_length is None:
            max_length = self.chunk_size

        if len(text) <= max_length:
            return [text]

        chunks = []
        # 문단 단위로 먼저 분리
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 문단 자체가 너무 길면 문장 단위로 분리
                if len(para) > max_length:
                    sentences = re.split(r'([.!?]\s+)', para)
                    current_chunk = ""
                    for i, sent in enumerate(sentences):
                        if len(current_chunk) + len(sent) <= max_length:
                            current_chunk += sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_documents(self, data_path: Path) -> List[DocumentChunk]:
        """모든 문서를 처리하여 청크 리스트 반환"""
        policy_data = self.load_policy_data(data_path)
        all_chunks = []

        for doc_key, doc_data in policy_data.items():
            sections = self.extract_sections(doc_data['content'])

            for sec in sections:
                # 섹션 내용을 청킹
                text_chunks = self.chunk_text(sec['content'])

                for i, text in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_key}_{sec['section']}_{i}",
                        content=text,
                        source=doc_data['source'],
                        doc_title=doc_data['title'],
                        doc_type=doc_data['type'],
                        section=sec['section'],
                        law_reference=doc_data['law_reference'],
                        effective_date=doc_data['effective_date']
                    )
                    all_chunks.append(chunk)

        return all_chunks

    def chunks_to_dict(self, chunks: List[DocumentChunk]) -> List[Dict]:
        """청크 리스트를 딕셔너리 리스트로 변환"""
        return [asdict(chunk) for chunk in chunks]

    def save_chunks(self, chunks: List[DocumentChunk], output_path: Path):
        """청크를 JSON 파일로 저장"""
        chunks_dict = self.chunks_to_dict(chunks)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, ensure_ascii=False, indent=2)
        print(f"청크 저장 완료: {output_path} ({len(chunks)}개 청크)")


def main():
    """문서 처리 실행"""
    base_dir = Path(__file__).parent.parent
    raw_data_path = base_dir / "data" / "raw" / "policy_data.json"
    processed_path = base_dir / "data" / "processed" / "chunks.json"

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
    chunks = processor.process_documents(raw_data_path)
    processor.save_chunks(chunks, processed_path)

    # 청크 통계 출력
    print(f"\n총 청크 수: {len(chunks)}")
    sources = {}
    for chunk in chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1
    print("출처별 청크 수:")
    for source, count in sources.items():
        print(f"  - {source}: {count}개")


if __name__ == "__main__":
    main()
