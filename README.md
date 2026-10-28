# Yardwise

컨테이너 재취급 감소를 위한 AI 이동 위치 추천 시뮬레이터입니다.

## 사용하기

[시뮬레이터 열기](https://zzongzzz47.github.io/yardwise/)

사례를 선택하고 출고를 진행하면서 AI 추천 위치와 규칙 기반 이동 결과를 비교할 수 있습니다. 데이터와 학습 모형을 포함한 정적 웹사이트로, 로그인이나 별도 설치 없이 브라우저에서 실행합니다.

## 분석 범위

공개 연구 자료의 컨테이너 배치와 출고 확률을 이용한 계산 실험입니다. 실제 항만의 실시간 운영 시스템이나 현장 검증 결과를 의미하지 않습니다. 출고 확률은 입력 자료로 제공되며, AI는 후보 목적지의 상대 비용을 학습해 이동 위치를 추천합니다.

## 데이터 출처

Zhang, Zhanluo (2024). *Probabilistic Container Relocation Problem*, Version 3. Mendeley Data. https://doi.org/10.17632/3j52cjrm52.3

원본 연구 데이터의 라이선스는 CC BY 4.0입니다. 본 프로젝트에서는 확률 정규화, 학습용 자료 생성, 모형 학습과 브라우저 실행용 변환을 수행했습니다.

## 파일 구성

- `index.html`, `styles.css`: 화면
- `app.js`, `engine.js`: 추천과 시뮬레이션
- `assets/data.json`: 시연 사례와 분석 결과
- `assets/model-*.bin`: 브라우저 추론용 학습 모형
- `.nojekyll`: GitHub Pages 정적 배포 설정
