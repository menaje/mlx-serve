---
type: work_plan
title: 'MLX-Serve v0.2: 성능 최적화, 운영 기능, 배포 개선'
reason: '프로덕션 환경에서의 성능과 운영성을 강화하고, 설치 편의성을 높여 사용자 확대 필요'
purpose: mlx-serve를 프로덕션 레벨의 안정적이고 고성능 임베딩/리랭킹 서버로 완성
summary: '설정 파일 지원, 배치 처리 최적화, Prometheus 메트릭, 모델 양자화, 자동 다운로드, Homebrew 배포'
tags:
  - mlx-serve
  - performance
  - monitoring
  - homebrew
  - quantization
category: feature
requires_update:
  - README.md
github_issue_number: 4
---

# MLX-Serve v0.2 Enhancement

## 1. 개요

### 1.1 배경
mlx-serve v0.1은 기본적인 임베딩/리랭킹 API와 서버 관리 기능을 제공합니다. v0.2에서는 프로덕션 환경에서 필요한 성능 최적화, 운영 기능, 배포 편의성을 강화합니다.

### 1.2 목표
1. **성능 최적화**: 배치 처리로 처리량 극대화
2. **운영 기능**: 메트릭, 설정 파일로 안정적 운영
3. **모델 유연성**: 양자화, 자동 다운로드로 사용성 향상
4. **배포 개선**: Homebrew로 손쉬운 설치

### 1.3 제미나이 협의 결과
- 설정 파일을 다른 기능의 기반으로 먼저 구현
- 배치 처리는 요청 단위 → 연속 배치 순으로 구현
- API 키 인증은 이번 버전에서 제외
- brew tap으로 macOS 사용자 접근성 강화

---

## 2. 작업 범위

### 2.1 Phase 1: 설정 파일 지원 (우선순위 1)

#### 2.1.1 설정 파일 위치
```
~/.mlx-serve/config.yaml
```

#### 2.1.2 설정 스키마
```yaml
# ~/.mlx-serve/config.yaml
server:
  host: "0.0.0.0"
  port: 8000

models:
  directory: ~/.mlx-serve/models
  preload:
    - Qwen3-Embedding-0.6B
  auto_download: true

cache:
  max_embedding_models: 3
  max_reranker_models: 2
  ttl_seconds: 1800

batch:
  max_batch_size: 32
  max_wait_ms: 50

metrics:
  enabled: true
  port: 9090

logging:
  level: INFO
  format: json  # text | json
```

#### 2.1.3 우선순위
1. 설정 파일 로드
2. 환경 변수 오버라이드
3. CLI 옵션 오버라이드

### 2.2 Phase 2: 배치 처리 최적화 (우선순위 2)

#### 2.2.1 요청 단위 배치
- 단일 요청 내 여러 입력을 MLX에서 병렬 처리
- 현재 순차 처리를 벡터화된 배치 처리로 변경

```python
# Before: 순차 처리
for text in inputs:
    embedding = model.encode(text)

# After: 배치 처리
embeddings = model.encode_batch(inputs)  # MLX 병렬 처리
```

#### 2.2.2 연속 배치 (Continuous Batching)
- 여러 요청을 큐에 모아 일괄 처리
- 설정 가능한 max_batch_size, max_wait_ms

```python
class BatchProcessor:
    def __init__(self, max_batch_size=32, max_wait_ms=50):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def process_loop(self):
        while True:
            batch = await self._collect_batch()
            results = self._process_batch(batch)
            self._distribute_results(batch, results)
```

### 2.3 Phase 3: Prometheus 메트릭 (우선순위 3)

#### 2.3.1 메트릭 엔드포인트
```
GET /metrics
```

#### 2.3.2 제공 메트릭
```python
# 요청 메트릭
mlx_serve_requests_total{endpoint="/v1/embeddings", status="success"}
mlx_serve_request_duration_seconds{endpoint="/v1/embeddings", quantile="0.99"}
mlx_serve_request_size_bytes{endpoint="/v1/embeddings"}

# 모델 메트릭
mlx_serve_model_load_seconds{model="Qwen3-Embedding-0.6B"}
mlx_serve_model_inference_seconds{model="Qwen3-Embedding-0.6B"}
mlx_serve_models_loaded{type="embedding"}

# 캐시 메트릭
mlx_serve_cache_hits_total{type="embedding"}
mlx_serve_cache_misses_total{type="embedding"}
mlx_serve_cache_evictions_total{type="embedding"}

# 시스템 메트릭
mlx_serve_memory_usage_bytes
mlx_serve_batch_size{endpoint="/v1/embeddings"}
```

#### 2.3.3 의존성
```toml
dependencies = [
    "prometheus-client>=0.17.0",
]
```

### 2.4 Phase 4: 모델 양자화 지원 (우선순위 4)

#### 2.4.1 양자화 옵션
```bash
# 모델 다운로드 시 양자화 적용
mlx-serve pull Qwen/Qwen3-Embedding-0.6B --quantize 4bit

# 기존 모델 양자화
mlx-serve quantize Qwen3-Embedding-0.6B --bits 4
```

#### 2.4.2 API 요청 시 양자화 모델 사용
```json
{
  "model": "Qwen3-Embedding-0.6B-4bit",
  "input": ["hello world"]
}
```

#### 2.4.3 메타데이터 확장
```json
{
  "name": "Qwen3-Embedding-0.6B",
  "quantization": "4bit",
  "original_size": 1200000000,
  "quantized_size": 350000000
}
```

### 2.5 Phase 5: 자동 모델 다운로드 (우선순위 5)

#### 2.5.1 동작 방식
1. API 요청 시 모델이 없으면 자동 다운로드 시도
2. HuggingFace에서 모델 검색 및 다운로드
3. 다운로드 중 요청은 대기 또는 에러 반환 (설정 가능)

#### 2.5.2 설정
```yaml
models:
  auto_download: true
  auto_download_timeout: 300  # 초
  auto_download_on_request: true  # false면 요청 시 에러 반환
```

#### 2.5.3 모델명 해석
```python
# 짧은 이름 → HuggingFace 저장소 매핑
MODEL_ALIASES = {
    "qwen-embedding": "Qwen/Qwen3-Embedding-0.6B",
    "qwen-reranker": "Qwen/Qwen3-Reranker-0.6B",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
```

### 2.6 Phase 6: 토큰 카운트 API (우선순위 6)

#### 2.6.1 엔드포인트
```
POST /v1/tokenize
```

#### 2.6.2 요청/응답
```json
// Request
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["Hello world", "How are you?"]
}

// Response
{
  "data": [
    {"index": 0, "tokens": 2, "token_ids": [1234, 5678]},
    {"index": 1, "tokens": 4, "token_ids": [1234, 5678, 9012, 3456]}
  ],
  "model": "Qwen3-Embedding-0.6B"
}
```

### 2.7 Phase 7: Homebrew 배포 (우선순위 7)

#### 2.7.1 저장소 구조
```
homebrew-mlx-serve/
├── Formula/
│   └── mlx-serve.rb
└── README.md
```

#### 2.7.2 Formula 예시
```ruby
class MlxServe < Formula
  include Language::Python::Virtualenv

  desc "MLX-based embedding and reranking server"
  homepage "https://github.com/menaje/mlx-serve"
  url "https://github.com/menaje/mlx-serve/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "..."
  license "MIT"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
  end

  service do
    run [opt_bin/"mlx-serve", "start", "--foreground"]
    keep_alive true
    log_path var/"log/mlx-serve.log"
    error_log_path var/"log/mlx-serve.error.log"
  end

  test do
    system "#{bin}/mlx-serve", "--version"
  end
end
```

#### 2.7.3 사용법
```bash
brew tap menaje/mlx-serve
brew install mlx-serve
brew services start mlx-serve
```

---

## 3. 기술 설계

### 3.1 파일 구조 변경
```
src/mlx_serve/
├── cli.py                    # quantize 명령 추가
├── config.py                 # YAML 설정 로더 추가
├── server.py                 # 메트릭 미들웨어 추가
├── core/
│   ├── batch_processor.py    # 새 파일: 배치 처리
│   ├── config_loader.py      # 새 파일: YAML 설정
│   ├── metrics.py            # 새 파일: Prometheus 메트릭
│   ├── model_manager.py      # 양자화, 자동 다운로드 추가
│   └── quantizer.py          # 새 파일: 양자화 유틸
└── routers/
    ├── embeddings.py         # 배치 처리 적용
    ├── rerank.py             # 배치 처리 적용
    └── tokenize.py           # 새 파일: 토큰 카운트 API
```

### 3.2 의존성 추가
```toml
[project]
dependencies = [
    # 기존 의존성...
    "prometheus-client>=0.17.0",
    "pyyaml>=6.0",
]
```

---

## 4. 구현 계획

### 4.1 단계별 작업

| 단계 | 작업 | 예상 파일 |
|------|------|----------|
| 1 | YAML 설정 파일 로더 구현 | `core/config_loader.py` |
| 2 | Settings 클래스 YAML 통합 | `config.py` |
| 3 | 배치 프로세서 구현 | `core/batch_processor.py` |
| 4 | 임베딩 라우터 배치 처리 적용 | `routers/embeddings.py` |
| 5 | 리랭커 라우터 배치 처리 적용 | `routers/rerank.py` |
| 6 | Prometheus 메트릭 모듈 구현 | `core/metrics.py` |
| 7 | 서버에 메트릭 미들웨어 추가 | `server.py` |
| 8 | 양자화 유틸 구현 | `core/quantizer.py` |
| 9 | CLI quantize 명령 추가 | `cli.py` |
| 10 | 자동 다운로드 로직 추가 | `core/model_manager.py` |
| 11 | 모델 앨리어스 구현 | `core/model_manager.py` |
| 12 | 토큰 카운트 API 구현 | `routers/tokenize.py` |
| 13 | Homebrew Formula 작성 | `homebrew-mlx-serve/` |
| 14 | 테스트 작성 | `tests/` |
| 15 | README.md 업데이트 | `README.md` |

### 4.2 테스트 계획
- 단위 테스트: 설정 로더, 배치 처리, 양자화
- 통합 테스트: API 엔드포인트, 메트릭
- 성능 테스트: 배치 처리 전후 처리량 비교

---

## 5. 리스크 및 고려사항

### 5.1 리스크
| 리스크 | 영향 | 대응 |
|--------|------|------|
| 배치 처리 복잡성 | 구현 지연 | 요청 단위 배치 먼저 완성 |
| 양자화 품질 저하 | 임베딩 정확도 감소 | 벤치마크 후 권장 설정 제공 |
| 자동 다운로드 지연 | 첫 요청 타임아웃 | 비동기 다운로드 + 상태 API |

### 5.2 하위 호환성
- 기존 환경 변수 계속 지원
- 설정 파일 없어도 기본값으로 동작
- 기존 API 응답 형식 유지

---

## 6. 완료 기준

- [ ] `~/.mlx-serve/config.yaml`로 서버 설정 가능
- [ ] 배치 처리로 처리량 2배 이상 향상
- [ ] `/metrics` 엔드포인트에서 Prometheus 메트릭 제공
- [ ] `mlx-serve quantize` 명령으로 모델 양자화 가능
- [ ] 모델 자동 다운로드 동작
- [ ] `/v1/tokenize` API 동작
- [ ] `brew install mlx-serve` 설치 가능
- [ ] 모든 테스트 통과
- [ ] README.md 업데이트
