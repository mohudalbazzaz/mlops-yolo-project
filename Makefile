fastapi:
	python3 -m uvicorn main:app --reload

streamlit:
	python3 -m streamlit run src/frontend/streamlit_app.py 

mlflow:
	mlflow ui --port 5000 --host 127.0.0.1

compose:
	docker compose up --build

pytest:
	python3 -m pytest

lint:
	python3 -m black . 