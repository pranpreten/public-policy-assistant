"""
공공 정책 질의응답 시스템 - Flask 웹 인터페이스
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import PolicyRAG

app = Flask(__name__)

# 전역 RAG 인스턴스
rag_system = None


def get_rag():
    """RAG 시스템 가져오기 (지연 로딩)"""
    global rag_system
    if rag_system is None:
        base_dir = Path(__file__).parent
        vectordb_path = str(base_dir / "vectordb")

        rag_system = PolicyRAG(
            vectordb_path=vectordb_path,
            llm_model="Qwen/Qwen2.5-14B-Instruct",
            llm_model_path=str(base_dir / "models" / "Qwen2.5-14B-Instruct")
        )
    return rag_system


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """질문 처리 API"""
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': '질문을 입력해주세요.'}), 400

    try:
        rag = get_rag()
        response = rag.answer(question)

        return jsonify({
            'answer': response.answer,
            'sources': response.sources,
            'question': response.question
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 서버 시작 전 모델 미리 로딩
    print("모델 사전 로딩 중...")
    get_rag()
    print("모델 로딩 완료! 서버 시작...")
    app.run(host='0.0.0.0', port=7865, debug=False)
