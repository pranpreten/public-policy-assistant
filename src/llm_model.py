"""
LLM 모델 모듈 (vLLM 버전)
- vLLM으로 빠른 추론
- GPU 최적화
- 로컬 모델 경로 지원
"""

from vllm import LLM, SamplingParams
from typing import Dict, Any
from pathlib import Path


class PolicyLLM:
    """공공 정책 질의응답용 LLM (vLLM)"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        model_path: str = None,  # 로컬 모델 경로 (선택)
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85
    ):
        """
        Args:
            model_name: HuggingFace 모델 이름 (model_path 없을 때 사용)
            model_path: 로컬 모델 경로 (있으면 우선 사용)
            tensor_parallel_size: GPU 병렬 수
            gpu_memory_utilization: GPU 메모리 사용률
        """
        # 로컬 경로가 있으면 로컬, 없으면 HuggingFace에서 다운로드
        if model_path and Path(model_path).exists():
            self.model_name = model_path
            print(f"vLLM 모델 로딩 (로컬): {model_path}")
        else:
            self.model_name = model_name
            print(f"vLLM 모델 로딩 (HuggingFace): {model_name}")

        # vLLM 모델 로드
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=8192  # GPU 메모리 제한으로 컨텍스트 길이 제한
        )

        print("vLLM 모델 로딩 완료")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """응답 생성"""

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        return response.strip()

    def create_qa_prompt(
        self,
        question: str,
        context_docs: list[Dict[str, Any]]
    ) -> str:
        """RAG 기반 QA 프롬프트 생성"""

        # 컨텍스트 문서들을 포맷팅
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            meta = doc['metadata']
            context_parts.append(f"""
[문서 {i}]
출처: {meta['source']} - {meta['doc_title']}
유형: {meta['doc_type']}
조항: {meta['section']}
관련법령: {meta['law_reference']}

내용:
{doc['content']}
""")

        context_text = "\n---\n".join(context_parts)

        # 시스템 프롬프트
        system_prompt = """당신은 대한민국 공공 정책 및 법령 전문가입니다.
사용자의 질문에 대해 제공된 참고 문서를 기반으로 정확하고 친절하게 답변해주세요.

답변 시 반드시 다음 규칙을 따라주세요:
1. 참고 문서에 있는 내용만을 기반으로 답변하세요.
2. 답변 마지막에 반드시 출처를 명시하세요.
3. 구체적인 기준(수치, 조건 등)이 있으면 정확히 언급하세요.
4. 관련 법령 조항이 있으면 함께 안내하세요."""

        # 전체 프롬프트 구성 (Qwen2.5 형식)
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
다음 참고 문서를 바탕으로 질문에 답변해주세요.

## 참고 문서
{context_text}

## 질문
{question}

## 답변 형식
답변 내용을 작성한 후, 마지막에 다음 형식으로 출처를 표시해주세요:
---
📋 출처: [출처 문서명]
📜 관련 법령: [관련 법령 조항]
📅 기준일: [시행일/기준일]<|im_end|>
<|im_start|>assistant
"""
        return prompt


def test_llm():
    """LLM 테스트"""
    print("=" * 60)
    print("vLLM 모델 테스트")
    print("=" * 60)

    # 모델 로드
    llm = PolicyLLM(
        model_name="Qwen/Qwen2.5-14B-Instruct"
    )

    # 간단한 테스트
    test_prompt = """<|im_start|>system
당신은 친절한 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
안녕하세요! 간단히 자기소개 해주세요.<|im_end|>
<|im_start|>assistant
"""

    print("\n테스트 질문: 안녕하세요! 간단히 자기소개 해주세요.")
    response = llm.generate_response(test_prompt, max_new_tokens=200)
    print(f"\n응답: {response}")


if __name__ == "__main__":
    test_llm()
