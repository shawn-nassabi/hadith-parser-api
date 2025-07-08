from fastapi import FastAPI
from app.routes import router

# to run: uvicorn main:app --reload

app = FastAPI(
    title="Hadith Parser API",
    description="Extracts sanad and matn from uploaded hadith files.",
    version="1.0.0"
)

app.include_router(router)