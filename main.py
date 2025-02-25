from fastapi import FastAPI
from routers.search_routers import router

app = FastAPI()

# Include the router
app.include_router(router)

# Load embedding model and tokenizer when the app starts


@app.on_event("startup")
async def startup_event():
    # This ensures the model and tokenizer are loaded when the app starts
    from services.search_services import model, tokenizer
    print("App started successfully!")
