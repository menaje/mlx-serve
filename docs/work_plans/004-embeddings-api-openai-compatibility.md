---
type: work_plan
title: 'Embeddings API OpenAI 호환성 개선: encoding_format 및 dimensions 파라미터 지원'
reason: OpenAI SDK 사용 시 encoding_format을 명시하지 않으면 차원 문제가 발생하여 완전한 drop-in 호환성이 필요
purpose: OpenAI SDK와의 완전한 호환성을 달성하여 사용자가 추가 설정 없이 mlx-serve를 사용할 수 있도록 함
summary: >-
  encoding_format(float/base64) 완전 지원, dimensions 파라미터 지원(슬라이싱+L2정규화), 테스트 케이스
  추가
tags:
  - mlx-serve
  - openai-compatibility
  - embeddings
  - api
category: feature
requires_update:
  - README.md
github_issue_number: 6
---

# Embeddings API OpenAI 호환성 개선

## 1. 개요

### 1.1 배경
mlx-serve의 임베딩 API를 OpenAI SDK로 호출할 때, `encoding_format`을 명시하지 않으면 "차원 문제"가 발생합니다. 현재 mlx-serve는:

1. `encoding_format` 파라미터를 선언만 하고 실제로 사용하지 않음
2. `base64` 형식 미지원 (항상 float 배열 반환)
3. `dimensions` 파라미터 미지원

### 1.2 목표
1. **encoding_format 완전 지원**: float/base64 형식 모두 지원
2. **dimensions 파라미터 지원**: Matryoshka 방식 임베딩 차원 축소
3. **OpenAI SDK 완전 호환**: 추가 설정 없이 drop-in 사용 가능

### 1.3 제미나이 협의 결과
- **방안 A (base64 완전 지원)** 채택
- base64 인코딩: float32 little-endian 형식 사용
- dimensions: 단순 슬라이싱이 아닌 **슬라이싱 + L2 정규화** 필요
- 네트워크 효율성: base64가 JSON float 배열 대비 25-35% 작음

---

## 2. 작업 범위

### 2.1 encoding_format 파라미터 완전 지원

#### 2.1.1 현재 상태
```python
class EmbeddingRequest(BaseModel):
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format of the embedding output",
    )

class EmbeddingData(BaseModel):
    embedding: list[float]  # 항상 float만 지원
```

#### 2.1.2 개선 후
```python
class EmbeddingData(BaseModel):
    embedding: list[float] | str  # float 또는 base64 문자열
```

#### 2.1.3 base64 인코딩 구현
OpenAI API 호환을 위해 **float32 little-endian** 형식 사용:

```python
import base64
import numpy as np

def encode_embedding_base64(embedding: list[float]) -> str:
    """임베딩을 base64로 인코딩 (OpenAI 호환 형식)."""
    # float32, little-endian 변환
    embedding_bytes = np.array(embedding, dtype="<f4").tobytes()
    return base64.b64encode(embedding_bytes).decode("utf-8")
```

### 2.2 dimensions 파라미터 지원

#### 2.2.1 요청 스키마 확장
```python
class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None  # 새로 추가
```

#### 2.2.2 차원 축소 구현
Matryoshka Representation Learning 방식: **슬라이싱 + L2 정규화**

```python
import numpy as np

def truncate_embedding(embedding: list[float], dimensions: int) -> list[float]:
    """임베딩 차원을 축소하고 정규화."""
    # 1. 앞 N개 차원 슬라이싱
    truncated = np.array(embedding[:dimensions], dtype=np.float32)

    # 2. L2 정규화 (벡터 크기를 1로)
    norm = np.linalg.norm(truncated)
    if norm > 0:
        truncated = truncated / norm

    return truncated.tolist()
```

#### 2.2.3 정규화가 필요한 이유
| 방식 | 벡터 크기 | 문제점 |
|------|----------|--------|
| 단순 슬라이싱 | 원본보다 작아짐 | 코사인 유사도 계산 왜곡 |
| 슬라이싱 + 정규화 | 항상 1 | 정확한 유사도 계산 |

### 2.3 응답 처리 흐름

```
요청 수신
    ↓
임베딩 생성 (모델)
    ↓
dimensions 파라미터 확인
    ├─ 있음 → 슬라이싱 + L2 정규화
    └─ 없음 → 원본 유지
    ↓
encoding_format 확인
    ├─ "float" → list[float] 반환
    └─ "base64" → base64 문자열 반환
    ↓
응답 반환
```

---

## 3. 기술 설계

### 3.1 파일 변경 목록

| 파일 | 변경 내용 |
|------|----------|
| `routers/embeddings.py` | EmbeddingRequest에 dimensions 추가, 응답 처리 로직 구현 |
| `tests/test_embeddings.py` | base64, dimensions 테스트 케이스 추가 |
| `README.md` | API 문서 업데이트 |

### 3.2 EmbeddingRequest 스키마 변경

```python
class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str = Field(..., description="Model name to use for embedding")
    input: str | list[str] = Field(..., description="Text(s) to embed")
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format of the embedding output",
    )
    dimensions: int | None = Field(
        default=None,
        description="Number of dimensions for the output embedding (requires MRL-trained model)",
        gt=0,
    )
```

### 3.3 EmbeddingData 스키마 변경

```python
class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str  # float 배열 또는 base64 문자열
    index: int
```

### 3.4 핵심 처리 로직

```python
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    # ... 기존 임베딩 생성 로직 ...

    # 후처리: dimensions + encoding_format
    processed_embeddings = []
    for emb in embeddings_list:
        # 1. dimensions 처리
        if request.dimensions and request.dimensions < len(emb):
            emb = truncate_embedding(emb, request.dimensions)

        # 2. encoding_format 처리
        if request.encoding_format == "base64":
            emb = encode_embedding_base64(emb)

        processed_embeddings.append(emb)

    # 응답 생성
    data = [
        EmbeddingData(embedding=emb, index=idx)
        for idx, emb in enumerate(processed_embeddings)
    ]
    # ...
```

---

## 4. 테스트 계획

### 4.1 단위 테스트

#### 4.1.1 base64 인코딩 테스트
```python
def test_encoding_format_base64():
    """Test base64 encoding format."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
            "encoding_format": "base64",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # base64 문자열인지 확인
    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, str)

    # 디코딩 후 원본 복원 검증
    decoded = np.frombuffer(
        base64.b64decode(embedding),
        dtype="<f4"
    )
    assert len(decoded) > 0
```

#### 4.1.2 dimensions 파라미터 테스트
```python
def test_dimensions_truncation():
    """Test dimensions parameter with truncation and normalization."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
            "dimensions": 256,
        },
    )
    assert response.status_code == 200
    data = response.json()

    embedding = data["data"][0]["embedding"]
    assert len(embedding) == 256

    # L2 정규화 검증 (벡터 크기 ≈ 1)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.001
```

#### 4.1.3 dimensions + base64 조합 테스트
```python
def test_dimensions_with_base64():
    """Test dimensions with base64 encoding."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
            "dimensions": 128,
            "encoding_format": "base64",
        },
    )
    assert response.status_code == 200

    embedding_b64 = response.json()["data"][0]["embedding"]
    decoded = np.frombuffer(base64.b64decode(embedding_b64), dtype="<f4")

    assert len(decoded) == 128
    assert abs(np.linalg.norm(decoded) - 1.0) < 0.001
```

### 4.2 경계 조건 테스트

| 테스트 케이스 | 예상 동작 |
|--------------|----------|
| dimensions > 원본 차원 | 원본 유지 (슬라이싱 안 함) |
| dimensions = 원본 차원 | 원본 유지 |
| dimensions = 1 | 1차원으로 축소 + 정규화 |
| dimensions = None | 원본 유지 |

---

## 5. 구현 순서

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | 헬퍼 함수 구현 (truncate_embedding, encode_embedding_base64) | `routers/embeddings.py` |
| 2 | EmbeddingRequest에 dimensions 필드 추가 | `routers/embeddings.py` |
| 3 | EmbeddingData.embedding 타입 변경 | `routers/embeddings.py` |
| 4 | create_embeddings 응답 처리 로직 구현 | `routers/embeddings.py` |
| 5 | 테스트 케이스 작성 | `tests/test_embeddings.py` |
| 6 | README.md 업데이트 | `README.md` |

---

## 6. 리스크 및 고려사항

### 6.1 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| MRL 미지원 모델에서 dimensions 사용 | 임베딩 품질 저하 | 문서에 주의사항 명시 |
| base64 인코딩 오버헤드 | 응답 지연 | numpy 벡터화로 최소화 |

### 6.2 하위 호환성
- 기존 API 요청 (encoding_format 미지정) → 기존과 동일하게 float 배열 반환
- dimensions 미지정 → 원본 차원 유지
- **기존 클라이언트 코드 변경 불필요**

---

## 7. 완료 기준

- [ ] `encoding_format: "base64"` 요청 시 올바른 base64 문자열 반환
- [ ] base64 디코딩 후 원본 float 배열 복원 가능
- [ ] `dimensions` 파라미터로 임베딩 차원 축소 가능
- [ ] 차원 축소 시 L2 정규화 적용 (벡터 크기 = 1)
- [ ] dimensions + base64 조합 정상 동작
- [ ] 모든 테스트 통과
- [ ] README.md API 문서 업데이트
