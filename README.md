# ELS 상품 리스크 연구

요구사항 v0.5 / 개발계획 v0.1을 기준으로 시작한 1인 연구 프로젝트입니다.
현재 구현 범위는 **데이터 파이프라인, ETC 변수선택, 실험 구간화, 52주 타겟,
코스피200 LightGBM 9개 모형 및 walk-forward 검증**입니다.
첫 모형은 분위수 보정이 부족하며 상품 위험확률과 대시보드는 아직 구현하지 않았습니다.

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
SHAP은 별도 `.[explain]` 옵션입니다.
핵심 의존성은 고정했으며 전체 전이 의존성 lock은 향후 Mac 환경 검증 후 추가합니다.

공식 참고: [FRED 관측 API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
[pandas 시간 집계](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).

## 52주 모형 학습 및 백테스트

위의 7변수 통합 배치를 만든 뒤 실행합니다. Mac에서 LightGBM 로딩 시 libomp 오류가 나면
LightGBM 공식 설치 안내에 따라 OpenMP 런타임을 설치해야 합니다.

```bash
python -m pip install -e '.[dev,model]'
python -m pytest -q
els experiment --batch data/research/combined --output artifacts/kospi200-week3 --target kospi200 --start 2020-01-03 --refit-weeks 13
```

기존 실험 파일을 덮어쓰지 않으므로 재실행 시 새 output 경로를 지정합니다.
52주 뒤 종가 수익률을 예측하며, 각 학습 기준일에 결과가 관측된 라벨만 학습합니다.
당주 변화, 1/4주 지연, 13주 평균/표준편차로 35개 후보를 만들고
전체 변수·ETC·ETC+실험 구간화를 같은 분할과 고정 하이퍼파라미터로 비교합니다.
실험 구간화에서 변동성 레벨 피처는 원값을 유지하며 해당 목록을 manifest에 기록합니다.
3주 이상 오래된 입력과 그 다음 주 변화값을 무효화하며, ETC 학습 구간 내부에
품질 결측이 있으면 조용히 이어붙이지 않고 실행을 거절합니다.

- report.md / metrics.json: 벤치마크·연도별 성능·분위수 커버리지
- predictions.csv: 예측일·라벨 종료일·실측·기본/분위수 예측·정렬 전 분위수
- folds.json: 매 fold의 학습 범위·라벨 관측 완료일·선택 변수·구간 경계
- models/: 최종 기본 모형 1개 + 분위수 모형 8개 + manifest.json
- forecast.json / latest_features.csv: 최종 기준일의 연구용 예측과 입력
- metadata.json: 입력 해시·실제 라이브러리 버전·학습 파라미터

모형 파일은 LightGBM 텍스트 형식입니다. `els.modeling.predict_bundle`로 재로딩합니다.
최종 예측은 연구용이며 아직 경보 확률로 사용할 수 없습니다.
현재 결과와 한계는 [3주차 결과](docs/week3-results.md)를 참고하세요.
