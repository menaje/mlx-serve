---
github_issue_number: 1
---
# mlx-serve MVP 작업계획서

## 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | mlx-serve |
| **목적** | Hugging Face 모델을 MLX로 변환하여 OpenAI 호환 API로 제공 |
| **실행 환경** | macOS 호스트 직접 실행 (Metal API 필요) |
| **배포 방식** | PyPI (1차) → Homebrew (2차) |

---

## 배경 및 제약사항

### Docker 실행 불가

| 구분 | 가능 여부 | 이유 |
|------|----------|------|
| Docker 내 MLX 실행 | ❌ | Linux 컨테이너가 Metal API에 접근 불가 |
| macOS 호스트 직접 실행 | ✅ | Metal 가속 사용 가능 |

### 설계 협의 결과 (Claude ↔ Gemini)

1. **MVP 범위**: 임베딩 + 리랭커 모두 포함
2. **모델 저장 경로**: `~/.mlx-serve/models` (환경변수 오버라이드 가능)
3. **API 전략**: "지능형 OpenAI 우선 (A+)" - OpenAI 호환 + Ollama 지능형 처리
4. **에러 형식**: OpenAI 호환 형식
5. **usage 정보**: 응답에 반드시 포함

---

## API 명세

### 데이터 플레인 (`/v1/*`) - OpenAI 호환

#### 1. 임베딩 API

```
POST /v1/embeddings
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["텍스트1", "텍스트2"],  // 단일 문자열도 자동 처리
  "encoding_format": "float"
}

→ 응답
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [...], "index": 0},
    {"object": "embedding", "embedding": [...], "index": 1}
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {"prompt_tokens": 15, "total_tokens": 15}
}
```

#### 2. 리랭킹 API (Jina 확장 형식)

```
POST /v1/rerank
{
  "model": "Qwen3-Reranker-0.6B",
  "query": "검색 쿼리",
  "documents": ["문서1", "문서2", "문서3"],
  "top_n": 3,
  "return_documents": true
}

→ 응답
{
  "results": [
    {"index": 2, "relevance_score": 0.98, "document": {"text": "문서3"}},
    {"index": 0, "relevance_score": 0.76, "document": {"text": "문서1"}}
  ],
  "usage": {"prompt_tokens": 45, "total_tokens": 45}
}
```

#### 3. 모델 목록 API

```
GET /v1/models

→ 응답
{
  "object": "list",
  "data": [
    {"id": "Qwen3-Embedding-0.6B", "object": "model", "owned_by": "mlx-serve", "type": "embedding"},
    {"id": "Qwen3-Reranker-0.6B", "object": "model", "owned_by": "mlx-serve", "type": "reranker"}
  ]
}
```

### 컨트롤 플레인 (`/api/*`) - Ollama 호환

#### 4. 모델 다운로드/변환

```
POST /api/pull
{
  "name": "Qwen/Qwen3-Embedding-0.6B",
  "type": "embedding"
}

→ 스트리밍 응답
{"status": "downloading", "completed": 50, "total": 100}
{"status": "converting"}
{"status": "success"}
```

#### 5. 모델 목록 (상세)

```
GET /api/tags

→ 응답
{
  "models": [
    {"name": "Qwen3-Embedding-0.6B", "size": 1200000000, "type": "embedding", "modified_at": "..."}
  ]
}
```

#### 6. 모델 삭제

```
DELETE /api/delete
{"name": "Qwen3-Embedding-0.6B"}
```

#### 7. 모델 상세 정보

```
POST /api/show
{"name": "Qwen3-Embedding-0.6B"}
```

### 에러 응답 형식 (OpenAI 호환)

```json
{
  "error": {
    "message": "Model 'invalid-model' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

---

## 프로젝트 구조

```
mlx-serve/
├── pyproject.toml              # 패키지 설정
├── src/
│   └── mlx_serve/
│       ├── __init__.py
│       ├── __main__.py         # python -m mlx_serve
│       ├── cli.py              # CLI (Typer)
│       ├── config.py           # 설정 관리 (Pydantic Settings)
│       ├── server.py           # FastAPI 앱
│       ├── core/
│       │   ├── __init__.py
│       │   └── model_manager.py  # 모델 로딩/변환/캐싱 (Singleton)
│       └── routers/
│           ├── __init__.py
│           ├── models.py       # /v1/models, /api/tags, /api/pull, /api/delete, /api/show
│           ├── embeddings.py   # /v1/embeddings
│           └── rerank.py       # /v1/rerank
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_rerank.py
│   └── test_models.py
└── README.md
```

---

## CLI 명령어

```bash
# 서버 실행
mlx-serve start [--port 8000] [--host 0.0.0.0]
mlx-serve start --foreground   # 포그라운드 실행

# 서비스 관리 (launchd)
mlx-serve service install      # 서비스 등록
mlx-serve service start        # 시작
mlx-serve service stop         # 중지
mlx-serve service status       # 상태 확인

# 모델 관리
mlx-serve pull Qwen/Qwen3-Embedding-0.6B --type embedding
mlx-serve pull Qwen/Qwen3-Reranker-0.6B --type reranker
mlx-serve list                 # 설치된 모델 목록
mlx-serve remove <model>       # 모델 삭제
```

---

## 기술 스택

| 용도 | 라이브러리 |
|------|-----------|
| API 프레임워크 | FastAPI + Uvicorn |
| CLI | Typer |
| 임베딩 | mlx-embeddings |
| 리랭커 | mlx-lm |
| 모델 다운로드 | huggingface_hub |
| 설정 관리 | Pydantic Settings |

---

## 구현 단계

### Phase 1: 프로젝트 초기화

- [ ] pyproject.toml 작성 (의존성, 메타데이터)
- [ ] 기본 프로젝트 구조 생성
- [ ] config.py 구현 (환경변수, 모델 경로)

### Phase 2: ModelManager 구현

- [ ] 모델 저장소 관리 (~/.mlx-serve/models)
- [ ] Hugging Face 모델 다운로드 기능
- [ ] MLX 변환 기능 (mlx-embeddings, mlx-lm 활용)
- [ ] 모델 메모리 캐싱 (Singleton)
- [ ] 모델 메타데이터 관리 (JSON)

### Phase 3: API 구현 - 임베딩

- [ ] POST /v1/embeddings 엔드포인트
- [ ] OpenAI 호환 요청/응답 스키마
- [ ] 단일 문자열 입력 자동 처리
- [ ] usage 정보 계산 및 포함
- [ ] 에러 핸들링

### Phase 4: API 구현 - 리랭킹

- [ ] POST /v1/rerank 엔드포인트
- [ ] Jina 확장 형식 (return_documents 옵션)
- [ ] usage 정보 계산 및 포함
- [ ] 에러 핸들링

### Phase 5: API 구현 - 모델 관리

- [ ] GET /v1/models (OpenAI 형식)
- [ ] POST /api/pull (스트리밍 응답)
- [ ] GET /api/tags (상세 목록)
- [ ] DELETE /api/delete
- [ ] POST /api/show

### Phase 6: CLI 구현

- [ ] mlx-serve start 명령어
- [ ] mlx-serve pull/list/remove 명령어
- [ ] mlx-serve service install/start/stop/status 명령어
- [ ] launchd plist 자동 생성

### Phase 7: 테스트 및 문서화

- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] README.md 작성
- [ ] PyPI 배포 준비

---

## 설정

### 환경변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `MLX_SERVE_HOST` | `0.0.0.0` | 서버 호스트 |
| `MLX_SERVE_PORT` | `8000` | 서버 포트 |
| `MLX_SERVE_MODELS_DIR` | `~/.mlx-serve/models` | 모델 저장 경로 |
| `MLX_SERVE_LOG_LEVEL` | `INFO` | 로그 레벨 |

### 모델 저장 구조

```
~/.mlx-serve/
├── models/
│   ├── Qwen3-Embedding-0.6B/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   └── Qwen3-Reranker-0.6B/
│       └── ...
└── metadata.json  # 설치된 모델 메타데이터
```

---

## 참고 자료

- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Jina Reranker API](https://jina.ai/reranker/)
