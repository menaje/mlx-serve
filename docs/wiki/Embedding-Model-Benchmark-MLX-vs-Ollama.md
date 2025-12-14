# Embedding Model Benchmark: MLX-Serve vs Ollama

본 문서는 동일한 Qwen3-Embedding-0.6B 모델을 MLX-Serve와 Ollama에서 각각 실행했을 때의 성능을 비교 분석한 결과입니다.

## 테스트 환경

| 항목 | 사양 |
|------|------|
| **Hardware** | Apple Silicon Mac |
| **OS** | macOS (Darwin 25.2.0) |
| **Date** | 2025-12-14 |

---

## 모델 사양 비교

### MLX-Serve

| 항목 | 값 |
|------|-----|
| **Model** | Qwen3-Embedding-0.6B |
| **Source** | Hugging Face ([Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) |
| **Framework** | MLX (Apple Silicon 최적화) |
| **Format** | SafeTensors (FP16) |
| **Size on Disk** | 1,151.6 MB |
| **Embedding Dimension** | 1024 |
| **Quantization** | None (FP16) |
| **Acceleration** | Metal GPU |

MLX-Serve는 Apple의 MLX 프레임워크를 사용하여 Metal GPU에서 직접 연산을 수행합니다. 모델은 FP16 (16-bit floating point) 정밀도로 실행되며, 양자화 없이 원본 가중치를 사용합니다.

### Ollama

| 항목 | 값 |
|------|-----|
| **Model** | qwen3-embedding:0.6b |
| **Source** | Ollama Registry |
| **Framework** | llama.cpp (GGML/GGUF) |
| **Format** | GGUF |
| **Size on Disk** | 639.2 MB |
| **Embedding Dimension** | 1024 |
| **Quantization** | Q8_0 (8-bit) |
| **Acceleration** | Metal GPU |

Ollama는 llama.cpp 기반으로 GGUF 포맷의 양자화된 모델을 사용합니다. Q8_0 양자화를 적용하여 모델 크기가 약 55% 감소했습니다.

### 사양 비교 요약

| 항목 | MLX-Serve | Ollama | 비고 |
|------|-----------|--------|------|
| **정밀도** | FP16 | Q8_0 | MLX가 더 높은 정밀도 |
| **모델 크기** | 1,151.6 MB | 639.2 MB | Ollama가 ~44% 작음 |
| **메모리 사용** | Higher | Lower | 양자화 효과 |
| **이론적 품질** | Higher | Slightly Lower | 양자화 손실 |

---

## 테스트 방법론

### 1. 테스트 설계

공정한 비교를 위해 다음 원칙을 적용했습니다:

1. **충분한 Warm-up**: 두 서비스 모두 10회의 사전 요청으로 모델 로딩 및 캐시 준비
2. **동일한 입력**: 모든 테스트에서 동일한 텍스트 사용
3. **다중 측정**: 각 테스트 10회 반복 후 평균 및 표준편차 산출
4. **다양한 시나리오**: 단일 텍스트와 배치 처리 모두 테스트

### 2. 테스트 케이스

#### A. 속도 테스트

**단일 텍스트 테스트**
```
"Machine learning is a subset of artificial intelligence
that enables systems to learn from data."
```

**배치 텍스트 테스트 (5개)**
```
1. "The quick brown fox jumps over the lazy dog."
2. "Artificial intelligence is transforming the world."
3. "Python is a popular programming language."
4. "Deep learning models require large amounts of data."
5. "Natural language processing enables computers to understand human language."
```

#### B. 품질 테스트 (Cosine Similarity)

의미적으로 유사한 문장 쌍과 비유사한 문장 쌍에 대한 코사인 유사도를 측정하여 임베딩 품질을 평가했습니다.

**유사 문장 쌍** (높은 유사도 기대)
| 문장 1 | 문장 2 |
|--------|--------|
| "I love programming in Python" | "Python is my favorite programming language" |
| "The weather is sunny today" | "It's a beautiful sunny day" |
| "Machine learning is fascinating" | "I find ML very interesting" |

**비유사 문장 쌍** (낮은 유사도 기대)
| 문장 1 | 문장 2 |
|--------|--------|
| "I love pizza" | "The stock market crashed yesterday" |
| "The cat is sleeping" | "Quantum physics is complex" |
| "Reading books is fun" | "The car needs an oil change" |

### 3. API 엔드포인트

**MLX-Serve**
```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": "text"}'
```

**Ollama**
```bash
curl http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding:0.6b", "input": ["text"]}'
```

---

## 테스트 결과

### 1. 속도 비교 (Latency)

#### 단일 텍스트 처리

| 측정 항목 | MLX-Serve | Ollama |
|-----------|-----------|--------|
| **평균** | 44.91 ms | 78.88 ms |
| **표준편차** | ±1.91 ms | ±1.83 ms |
| **최소** | 41.19 ms | 76.62 ms |
| **최대** | 47.26 ms | 83.17 ms |

**결과: MLX-Serve가 1.76배 빠름**

#### 배치 처리 (5개 텍스트)

| 측정 항목 | MLX-Serve | Ollama |
|-----------|-----------|--------|
| **평균** | 52.57 ms | 200.55 ms |
| **표준편차** | ±3.10 ms | ±3.35 ms |
| **최소** | 47.90 ms | 192.68 ms |
| **최대** | 55.89 ms | 204.96 ms |

**결과: MLX-Serve가 3.82배 빠름**

#### 처리량 (Throughput)

| 처리 방식 | MLX-Serve | Ollama | 비율 |
|-----------|-----------|--------|------|
| **단일 요청** | 22.3 texts/sec | 12.7 texts/sec | 1.76x |
| **배치 요청** | 95.1 texts/sec | 24.9 texts/sec | 3.82x |

### 2. 품질 비교 (Embedding Quality)

#### 유사 문장 쌍 코사인 유사도 (높을수록 좋음)

| 문장 쌍 | MLX-Serve | Ollama |
|---------|-----------|--------|
| Python programming | 0.8552 | 0.8538 |
| Sunny weather | 0.8775 | 0.8786 |
| Machine learning | 0.7786 | 0.7836 |
| **평균** | **0.8371** | **0.8387** |

#### 비유사 문장 쌍 코사인 유사도 (낮을수록 좋음)

| 문장 쌍 | MLX-Serve | Ollama |
|---------|-----------|--------|
| Pizza vs Stock market | 0.4366 | 0.4324 |
| Cat vs Quantum physics | 0.3158 | 0.3148 |
| Books vs Car | 0.4661 | 0.4657 |
| **평균** | **0.4062** | **0.4043** |

#### 변별력 (Discrimination Ability)

변별력 = 유사 문장 평균 유사도 - 비유사 문장 평균 유사도

| 서비스 | 변별력 |
|--------|--------|
| MLX-Serve | 0.4310 |
| Ollama | 0.4344 |

**결과: 품질 차이는 1% 미만으로 실질적으로 동등**

---

## 결과 분석

### 속도 측면

```
┌─────────────────────────────────────────────────────────────┐
│                    Latency Comparison                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Single Text:                                                │
│  MLX-Serve  ████████████████████ 44.91ms                    │
│  Ollama     ███████████████████████████████████ 78.88ms     │
│                                                              │
│  Batch (5):                                                  │
│  MLX-Serve  ██████████████ 52.57ms                          │
│  Ollama     █████████████████████████████████████████████   │
│             ██████████████████████████████ 200.55ms         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

1. **단일 텍스트**: MLX-Serve가 약 34ms 빠름 (1.76x)
2. **배치 처리**: MLX-Serve가 약 148ms 빠름 (3.82x)
3. **배치 효율성**: MLX-Serve는 배치 크기가 증가해도 처리 시간이 크게 늘지 않음

MLX 프레임워크의 Metal GPU 최적화가 효과적으로 작동하며, 특히 배치 처리에서 우수한 성능을 보입니다.

### 품질 측면

| 평가 지표 | MLX-Serve | Ollama | 차이 |
|-----------|-----------|--------|------|
| 유사 문장 유사도 | 0.8371 | 0.8387 | -0.19% |
| 비유사 문장 유사도 | 0.4062 | 0.4043 | +0.47% |
| 변별력 | 0.4310 | 0.4344 | -0.78% |

- Q8_0 양자화에 의한 품질 손실이 거의 없음
- 두 서비스 모두 유사/비유사 문장을 명확히 구분
- 실용적 관점에서 품질 차이는 무시 가능

---

## 결론

### 종합 평가

| 항목 | MLX-Serve | Ollama | 승자 |
|------|-----------|--------|------|
| **단일 요청 속도** | 44.91ms | 78.88ms | MLX-Serve |
| **배치 처리 속도** | 52.57ms | 200.55ms | MLX-Serve |
| **처리량 (배치)** | 95.1 t/s | 24.9 t/s | MLX-Serve |
| **임베딩 품질** | 0.8371 | 0.8387 | 동등 |
| **모델 크기** | 1,151.6 MB | 639.2 MB | Ollama |
| **메모리 효율** | Lower | Higher | Ollama |

### 권장 사용 시나리오

**MLX-Serve 권장:**
- 대량의 텍스트 임베딩 처리
- 배치 처리가 주된 워크로드
- 최대 처리량이 중요한 경우
- Apple Silicon Mac에서 최고 성능 필요

**Ollama 권장:**
- 메모리가 제한된 환경
- 다양한 모델을 동시에 로드해야 하는 경우
- 간편한 모델 관리가 필요한 경우
- 크로스 플랫폼 지원 필요

### 최종 결론

**MLX-Serve는 Apple Silicon에서 Ollama 대비 1.76x~3.82x 빠른 속도**를 제공하면서 **동등한 임베딩 품질**을 유지합니다. 대량의 텍스트를 처리해야 하는 RAG 시스템이나 임베딩 파이프라인에서는 MLX-Serve가 더 적합합니다.

---

## 부록: 테스트 코드

<details>
<summary>Python 테스트 스크립트</summary>

```python
import time
import requests
import numpy as np
from statistics import mean, stdev

MLX_URL = "http://localhost:8000/v1/embeddings"
OLLAMA_URL = "http://localhost:11434/api/embed"

def mlx_embed(texts):
    start = time.perf_counter()
    resp = requests.post(MLX_URL, json={
        "model": "Qwen3-Embedding-0.6B",
        "input": texts if isinstance(texts, list) else [texts]
    })
    elapsed = time.perf_counter() - start
    data = resp.json()
    embeddings = [d["embedding"] for d in data["data"]]
    return embeddings, elapsed

def ollama_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    start = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json={
        "model": "qwen3-embedding:0.6b",
        "input": texts
    })
    elapsed = time.perf_counter() - start
    data = resp.json()
    embeddings = data.get("embeddings", [])
    return embeddings, elapsed

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Warm-up
for i in range(10):
    mlx_embed("warm up text")
    ollama_embed("warm up text")

# Test execution
# ... (see full test code in repository)
```

</details>

---

*Last updated: 2025-12-14*
