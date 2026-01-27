#!/usr/bin/env python3
"""
공공 정책 질의응답 시스템 실행 스크립트

사용법:
    python run.py --mode web      # 웹 서버 실행
    python run.py --mode cli      # CLI 모드 실행
    python run.py --mode setup    # 데이터 및 벡터 DB 설정
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))


def setup():
    """데이터 수집 및 벡터 DB 구축"""
    print("=" * 60)
    print("데이터 설정 시작")
    print("=" * 60)

    # 1. 데이터 수집
    print("\n[1/3] 데이터 수집...")
    from scripts.collect_data import save_policy_data
    save_policy_data()

    # 2. 문서 청킹
    print("\n[2/3] 문서 청킹...")
    from src.document_processor import main as process_docs
    process_docs()

    # 3. 벡터 DB 구축
    print("\n[3/3] 벡터 DB 구축...")
    from src.vector_store import build_vector_store
    build_vector_store()

    print("\n" + "=" * 60)
    print("설정 완료!")
    print("=" * 60)


def run_cli():
    """CLI 모드 실행"""
    from src.rag_pipeline import PolicyRAG

    base_dir = Path(__file__).parent
    vectordb_path = str(base_dir / "vectordb")

    print("=" * 60)
    print("공공 정책 질의응답 시스템 (CLI)")
    print("=" * 60)
    print("종료: 'quit' 또는 'exit' 입력\n")

    rag = PolicyRAG(
        vectordb_path=vectordb_path,
        llm_model="Qwen/Qwen2.5-14B-Instruct",
        llm_model_path=str(base_dir / "models" / "Qwen2.5-14B-Instruct")
    )

    while True:
        try:
            question = input("\n❓ 질문: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("종료합니다.")
                break

            if not question:
                continue

            response = rag.answer(question)
            print(rag.format_response(response))

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break


def run_web():
    """웹 서버 실행"""
    from app import app, get_rag
    print("=" * 60)
    print("공공 정책 질의응답 시스템 (Web)")
    print("=" * 60)
    print("모델 사전 로딩 중...")
    get_rag()
    print("모델 로딩 완료!")
    print(f"서버 주소: http://0.0.0.0:7865")
    print("=" * 60)
    app.run(host='0.0.0.0', port=7865, debug=False)


def main():
    parser = argparse.ArgumentParser(description='공공 정책 질의응답 시스템')
    parser.add_argument(
        '--mode',
        choices=['web', 'cli', 'setup'],
        default='web',
        help='실행 모드 (web: 웹서버, cli: 명령줄, setup: 초기설정)'
    )

    args = parser.parse_args()

    if args.mode == 'setup':
        setup()
    elif args.mode == 'cli':
        run_cli()
    else:
        run_web()


if __name__ == '__main__':
    main()
