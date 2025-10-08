.PHONY: setup train train_xgb app test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	python -m src.train_lr

train_xgb:
	python -m src.train_xgb

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q
