"""Small reproducible Korean decision evaluation using official mlx-vlm predict.

Usage: python scripts/evaluate_decisions.py MODEL_PATH [OUTPUT_JSON]
"""

import json
import resource
import sys
import time
from pathlib import Path

CASES = [
    {
        "id": "document_type",
        "state": "제목: 결제 환불 요청. 결제가 두 번 청구되어 한 건의 환불을 요청합니다.",
        "question": {
            "type": "choice",
            "instructions": "이 문서의 유형을 선택하세요.",
            "criteria": {"support": "고객 문의", "minutes": "회의록", "receipt": "영수증"},
        },
        "expected": "support",
    },
    {
        "id": "note_topic",
        "state": "다음 주 화요일까지 새 앱 기능을 개발하고 테스트한 뒤 배포한다.",
        "question": {
            "type": "choice",
            "instructions": "메모의 주제를 선택하세요.",
            "criteria": {"payroll": "급여", "development": "소프트웨어 개발", "travel": "출장"},
        },
        "expected": "development",
    },
    {
        "id": "task_route",
        "state": "로그인이 안 되고 오류 화면이 나타나므로 기술 담당자의 도움이 필요합니다.",
        "question": {
            "type": "choice",
            "instructions": "어느 팀이 처리해야 하나요?",
            "criteria": {"billing": "결제 및 환불", "technical": "시스템 오류 해결"},
        },
        "expected": "technical",
    },
    {
        "id": "condition_true",
        "state": "계약은 60일 후 만료된다. 갱신 신청은 만료 90일 이내에 할 수 있다.",
        "question": {"type": "noul", "instructions": "지금 갱신 신청이 가능한가요?"},
        "expected": True,
    },
    {
        "id": "condition_false",
        "state": "물건은 배송 완료로 표시되었지만 고객은 아직 받지 못했다고 문의했다.",
        "question": {"type": "noul", "instructions": "고객이 환불을 요청했나요?"},
        "expected": False,
    },
    {
        "id": "priority",
        "state": (
            "전체 고객이 서비스에 접속할 수 없어 업무가 중단되었습니다. 즉시 복구가 필요합니다."
        ),
        "question": {
            "type": "score",
            "instructions": "이 문제의 긴급도는 어느 정도인가요?",
            "criteria": ["낮음", "보통", "높음"],
        },
        "expected": 2,
    },
]


def main():
    import mlx.core as mx
    from mlx_vlm import load, predict

    model_path = Path(sys.argv[1])
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, processor = load(str(model_path), trust_remote_code=False, strict=True)
    load_seconds = time.perf_counter() - started
    results = []
    for case in CASES:
        started = time.perf_counter()
        output = predict(model, processor, case["state"], {"q": case["question"]})
        elapsed = time.perf_counter() - started
        answer = output["answers"]["q"]
        predicted = (
            max(answer["probabilities"], key=answer["probabilities"].get)
            if case["question"]["type"] == "score"
            else answer["value"]
        )
        if case["question"]["type"] == "score":
            predicted = int(predicted)
        results.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "predicted": predicted,
                "correct": predicted == case["expected"],
                "answer": answer,
                "elapsed_seconds": round(elapsed, 4),
                "usage": output["usage"],
            }
        )
    result = {
        "model_path": str(model_path),
        "architecture": getattr(model, "model_type", model.config.model_type),
        "load_seconds": round(load_seconds, 4),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "correct": sum(x["correct"] for x in results),
        "total": len(results),
        "cases": results,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
