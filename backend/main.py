from fastapi import FastAPI
from app.api.auth import auth as auth_router
from app.api.users import users as users_router

app = FastAPI(title="PS-87 Backend")

app.include_router(auth_router.router)
app.include_router(users_router.router)


@app.get("/")
def root():
    return {"message": "PS-87 Backend API"}