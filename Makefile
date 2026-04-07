fastapi:
	python -m uvicorn main:app --reload

streamlit:
	python -m streamlit run src/frontend/streamlit_app.py 

mlflow:
	mlflow ui --port 5000 --host 127.0.0.1

compose:
	docker compose up --build

pytest:
	python -m pytest