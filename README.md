# ELS 상품 리스크 연구

요구사항 v0.5 / 개발계획 v0.1을 기준으로 시작한 1인 연구 프로젝트입니다.
현재 구현 범위는 **1주차 데이터 파이프라인(FR-01~03)**입니다.
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

FRED 예제는 미국 10년 금리와 원/달러 환율입니다. 5대 지수 데이터 확보는 별도입니다.
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

모형·앱 의존성은 이후 `pip install -e '.[model,app]'`로 설치할 수 있습니다.
핵심 의존성은 고정했으며 전체 전이 의존성 lock은 향후 Mac 환경 검증 후 추가합니다.

공식 참고: [FRED 관측 API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
[pandas 시간 집계](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).
