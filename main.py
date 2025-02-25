from fastapi import FastAPI
from routers.search_routers import router

app = FastAPI()

# Include the router
app.include_router(router)
