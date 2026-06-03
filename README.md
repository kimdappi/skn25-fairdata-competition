# 공정거래위원회 공모전

### [Track 2] 공정거래 공공데이터와 AI활용 인공지능 모델 개발

이 저장소는 `공정위AI공모전 모델제출 가이드.md` 기준으로 제출 가능한 Docker 패키지 형태로 정리되어 있습니다.

## 현재 구조

```text
.
├── app/
├── docs/
├── index/
├── models/
├── raw/
├── scripts/
├── Dockerfile
├── requirements.txt
└── server.py
```

## 실행

```bash
python3 scripts/validate_submission.py
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 제출 이미지 생성

```bash
bash scripts/build_submission.sh
```

자세한 내용은 [PROJECT_SETUP.md](/home/ming9/skn25-fairdata-competition/docs/PROJECT_SETUP.md) 와 [SUBMISSION_CHECKLIST.md](/home/ming9/skn25-fairdata-competition/docs/SUBMISSION_CHECKLIST.md) 를 보면 됩니다.
