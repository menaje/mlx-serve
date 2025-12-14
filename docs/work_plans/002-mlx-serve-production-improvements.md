---
type: work_plan
title: 'MLX-Serve 프로덕션 개선: 모델 관리, 서버 제어, API 확장'
reason: 현재 mlx-serve는 기본적인 기능만 구현되어 있어 프로덕션 환경에서의 안정성과 사용성이 부족함
purpose: 프로덕션 레벨의 안정적인 서버 운영을 위한 핵심 기능 구현
summary: '모델 캐시 관리(LRU/TTL), 서버 실행/종료 제어(PID 기반), 리랭커 API 확장(텍스트 출력) 구현'
tags:
  - mlx-serve
  - production
  - model-management
  - server-control
  - api
category: feature
requires_update:
  - README.md
github_issue_number: 2
---

# MLX-Serve 프로덕션 개선

## 1. 개요

### 1.1 배경
mlx-serve v0.1.0은 기본적인 임베딩/리랭킹 API를 제공하지만, 프로덕션 환경에서 필요한 다음 기능들이 부족합니다:
- 메모리 관리 없이 모델이 무한히 누적
- 서버 종료 명령 미지원 (background 모드)
- 리랭커 텍스트 출력 미지원

### 1.2 목표
1. **안정적인 메모리 관리**: LRU + TTL 기반 모델 캐시
2. **서버 라이프사이클 제어**: start/stop/status 명령
3. **API 확장성**: 리랭커 텍스트 출력 옵션

### 1.3 제미나이 협의 결과
- LRU + TTL 조합 캐시 전략 합의
- 기존 API에 옵션 추가 방식 합의
- PID 파일 기반 서버 관리 합의

---

## 2. 작업 범위

### 2.1 Phase 1: 서버 실행/종료 제어 (우선순위 1)

#### 2.1.1 PID 파일 관리
- 위치: `~/.mlx-serve/mlx-serve-<port>.pid`
- 서버 시작 시 PID 파일 생성
- 서버 종료 시 PID 파일 삭제
- 다중 인스턴스 지원 (포트별 구분)

#### 2.1.2 새로운 CLI 명령
```bash
# 서버 상태 확인
mlx-serve status [--port PORT]

# 서버 종료
mlx-serve stop [--port PORT] [--force]
```

#### 2.1.3 Graceful Shutdown
- SIGTERM 핸들러 등록
- 진행 중인 요청 완료 대기
- PID 파일 정리

### 2.2 Phase 2: 모델 캐시 관리 (우선순위 2)

#### 2.2.1 LRU 캐시 도입
```python
from cachetools import LRUCache

class ModelCache:
    embeddings: LRUCache  # maxsize=3
    rerankers: LRUCache   # maxsize=2
```

#### 2.2.2 TTL 기반 자동 언로드
- 기본 TTL: 30분 (설정 가능)
- 미사용 모델 자동 메모리 해제

#### 2.2.3 설정 옵션
```python
class Settings:
    cache_max_embedding_models: int = 3
    cache_max_reranker_models: int = 2
    cache_ttl_seconds: int = 1800
```

### 2.3 Phase 3: 리랭커 API 확장 (우선순위 3)

#### 2.3.1 요청 파라미터 추가
```python
class RerankRequest:
    return_text: bool = False
    decision_threshold: float = 0.5
```

#### 2.3.2 응답 필드 추가
```python
class RerankResult:
    text_output: str | None  # "yes" or "no"
```

### 2.4 Phase 4: 프리로드 옵션 (우선순위 4)

#### 2.4.1 CLI 옵션
```bash
mlx-serve start --preload MODEL_NAME
```

#### 2.4.2 설정 파일 지원
```yaml
# ~/.mlx-serve/config.yaml
preload_models:
  - Qwen3-Embedding-0.6B
```

### 2.5 Phase 5: Linux systemd 서비스 지원 (우선순위 5)

#### 2.5.1 개요
- macOS: launchd 지원 (이미 구현됨)
- Linux: systemd user service 추가
- Windows: 미지원 (대상 외)

#### 2.5.2 Linux systemd user service
```ini
# ~/.config/systemd/user/mlx-serve.service
[Unit]
Description=MLX-Serve Embedding Server
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/mlx-serve start --foreground
Restart=on-failure
RestartSec=5
Environment=PATH=%h/.local/bin:/usr/bin

[Install]
WantedBy=default.target
```

#### 2.5.3 CLI 명령 확장
```bash
# 자동 OS 감지
mlx-serve service install   # macOS: launchd, Linux: systemd
mlx-serve service start
mlx-serve service stop
mlx-serve service status

# 로그인 시 자동 시작 설정
mlx-serve service enable    # 로그인 시 자동 시작 활성화
mlx-serve service disable   # 로그인 시 자동 시작 비활성화
```

#### 2.5.4 설정 파일 지원
```yaml
# ~/.mlx-serve/config.yaml
service:
  auto_start: true          # 로그인 시 자동 시작 (기본값: false)
  restart_on_failure: true  # 크래시 시 자동 재시작
  restart_delay: 5          # 재시작 대기 시간 (초)
```

#### 2.5.5 구현 방식
- `platform.system()`으로 OS 감지
- macOS → 기존 LaunchdManager 사용 (RunAtLoad 옵션)
- Linux → 새로운 SystemdManager 구현 (WantedBy 옵션)
- `enable/disable` 명령으로 auto_start 토글

---

## 3. 기술 설계

### 3.1 파일 구조 변경
```
src/mlx_serve/
├── cli.py              # stop, status 명령 추가
├── config.py           # 캐시 설정 추가
├── server.py           # shutdown 핸들러 추가
├── core/
│   ├── model_manager.py  # LRU/TTL 캐시 적용
│   ├── pid_manager.py    # 새 파일: PID 관리
│   └── service_manager.py  # 새 파일: OS별 서비스 관리
└── routers/
    └── rerank.py         # return_text 옵션 추가
```

### 3.2 의존성 추가
```toml
[project]
dependencies = [
    # 기존 의존성...
    "cachetools>=5.0.0",
]
```

### 3.3 설정 스키마
```python
class Settings(BaseSettings):
    # 기존 설정...

    # 캐시 설정
    cache_max_embedding_models: int = 3
    cache_max_reranker_models: int = 2
    cache_ttl_seconds: int = 1800

    # 프리로드 설정
    preload_models: list[str] = []
```

---

## 4. 구현 계획

### 4.1 단계별 작업

| 단계 | 작업 | 예상 파일 |
|------|------|----------|
| 1 | PID 관리 모듈 구현 | `core/pid_manager.py` |
| 2 | CLI stop/status 명령 추가 | `cli.py` |
| 3 | Graceful shutdown 구현 | `server.py` |
| 4 | LRU 캐시 적용 | `model_manager.py` |
| 5 | TTL 캐시 적용 | `model_manager.py` |
| 6 | 캐시 설정 추가 | `config.py` |
| 7 | 리랭커 텍스트 출력 | `routers/rerank.py` |
| 8 | 프리로드 옵션 구현 | `cli.py`, `server.py` |
| 9 | Linux systemd 서비스 관리 | `core/service_manager.py`, `cli.py` |
| 10 | 테스트 작성 | `tests/` |
| 11 | 문서 업데이트 | `README.md` |

### 4.2 테스트 계획
- 단위 테스트: PID 관리, 캐시 동작
- 통합 테스트: 서버 시작/종료 플로우
- E2E 테스트: API 응답 검증

---

## 5. 리스크 및 고려사항

### 5.1 리스크
| 리스크 | 영향 | 대응 |
|--------|------|------|
| TTL 캐시 타이밍 이슈 | 요청 중 모델 언로드 | 요청 처리 중 TTL 리셋 |
| PID 파일 잔존 | 서버 시작 실패 | stale PID 파일 감지 및 정리 |
| Graceful shutdown 타임아웃 | 서버 종료 지연 | 타임아웃 설정 (기본 30초) |

### 5.2 하위 호환성
- 기존 API 변경 없음 (옵션 추가만)
- 기존 CLI 명령 동작 유지
- 기본값으로 현재 동작 유지

---

## 6. 완료 기준

- [ ] `mlx-serve stop` 명령으로 서버 종료 가능
- [ ] `mlx-serve status` 명령으로 상태 확인 가능
- [ ] LRU 캐시로 모델 수 제한 동작
- [ ] TTL 후 미사용 모델 자동 언로드
- [ ] 리랭커 API에서 `return_text` 옵션 동작
- [ ] 프리로드 옵션으로 서버 시작 시 모델 로드
- [ ] Linux에서 `mlx-serve service install/start/stop` 동작
- [ ] `mlx-serve service enable/disable`로 자동 시작 설정 가능
- [ ] 모든 테스트 통과
- [ ] README.md 업데이트
