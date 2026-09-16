# ELS 상품 리스크 연구

요구사항 v0.5 / 개발계획 v0.1을 기준으로 시작한 1인 연구 프로젝트입니다.
현재 구현 범위는 **데이터 파이프라인(FR-01~03), ETC 변수선택(FR-05), 실험 구간화(FR-04 일부)**입니다.
예측 모형, 상품 위험확률, 대시보드는 아직 구현하지 않았습니다.

## 실행 (Mac / Python 3.11+)

```bash
git clone https://github.com/YUJAEYUN/ELS.git
cd ELS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
els demo
```

Windows PowerShell에서는 가상환경 활성화 대신
`.venv\Scripts\python.exe -m pip install -r requirements.txt`처럼 실행할 수 있습니다.

데모는 고정 시드의 **합성 데이터**입니다. 실제 지수 성과가 아닙니다.
`data/demo/result/`에 weekly.csv, quality.csv, metadata.json, 정규화한 원주기 CSV가 생성됩니다.

## 실제 데이터

FRED 키를 환경변수로 설정한 뒤 실행합니다. 키와 데이터는 커밋하지 않습니다.
`.env.example`은 참고용이며 자동으로 읽지 않습니다.

```bash
export FRED_API_KEY='발급받은 키'
els batch --config config/fred.example.json --as-of 2026-09-11 --output data/runs/2026-09-11
```

FRED 예제는 미국 10년 금리와 원/달러 환율입니다. 5대 지수는 아래 실험 수집 경로를 지원합니다.
자신이 사용권을 가진 CSV도 `date,value` 컬럼으로 입력할 수 있습니다.
설정의 path는 설정 파일 폴더 기준입니다. source=csv, transform=level/return/difference를 지원합니다.
금리처럼 0 또는 음수가 가능한 변수는 비율 수익률 대신 difference를 사용합니다.

금요일 날짜까지 공개된 관측을 사용하며 다음 주 배치를 전제로 합니다.
미완료 주는 제외하고 선행 결측은 그대로 둡니다. LOCF 값에는 관측일과 경과일,
3주 이상 최신성 경고를 남깁니다. availability_lag_days는 달력일 기준입니다.
실제 발표 지연은 소스별 검증이 필요합니다. 현재 빈티지 데이터는 엄밀한 과거 시점 백테스트용이 아닙니다.

API 실패는 재시도 후 배치를 실패시킵니다. 이전 실행 캐시 대체·알림·연속 실패 이력은 7주차 과제입니다.
동일 output 경로를 다시 쓰면 덮어쓰므로 실행별 별도 경로를 사용합니다.

## 구조

- src/els: 소스 어댑터, 주간 전처리, 배치 CLI
- config: 소스 설정 예제
- tests: 시간 정합·결측·입력 검증·어댑터·데모 통합 테스트
- docs: 요구사항 정리, 8주 개발계획, 설계 검토 항목

## 5대 지수 수집과 ETC 선택

공개 차트 경로(Naver 코스피200, Yahoo 해외 4지수)를 사용합니다. API 키는 필요 없습니다.
1992년까지 모든 지수를 제공하지 않으며 소스별 범위는 docs/week2-results.md에 기록했습니다.
로컬에 저장된 자료는 연구용이며 저장소에 포함하지 않습니다.

```bash
els batch --config config/market.example.json --as-of 2026-09-11 --output data/runs/indices-2026-09-11
# 위 지수 배치와 앞의 FRED 배치를 먼저 실행한 뒤 저장 CSV로 재생
els batch --config config/research.snapshot.example.json --as-of 2026-09-11 --output data/research/combined
els features --input data/research/combined/weekly.csv --config config/research.snapshot.example.json --train-end 2023-12-29 --output data/research/features --experimental-bins
```

selection.json에 점수·선택 변수·훈련 기간·구간 경계·입력 해시를 저장합니다.
selected.csv는 훈련에서 고정한 선택/경계를 전체 기간에 적용한 결과이며 훈련 전용 파일이 아닙니다.
`--experimental-bins`를 생략하면 구간화하지 않은 선택 변수 원값을 보존합니다.
실험 구간화는 변수 자기분포 ES에 기반한 순서 구간이며 타겟 단조성을 보장하지 않습니다.
52주 타겟 학습 시점의 라벨 성숙 조건은 추후 모델링 단계에서 별도로 적용해야 합니다.

모형·앱 의존성은 이후 `pip install -e '.[model,app]'`로 설치할 수 있습니다.
핵심 의존성은 고정했으며 전체 전이 의존성 lock은 향후 Mac 환경 검증 후 추가합니다.

공식 참고: [FRED 관측 API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
[pandas 시간 집계](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).
