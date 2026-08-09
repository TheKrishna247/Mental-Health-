import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.db.database import get_db
from app.db.models import Base, User

TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_signup_success():
    response = client.post("/api/auth/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "test123"
    })
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "created_at" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_signup_duplicate_email():
    client.post("/api/auth/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "test123"
    })

    response = client.post("/api/auth/signup", json={
        "name": "Another User",
        "email": "test@example.com",
        "password": "test456"
    })
    assert response.status_code == 409


def test_login_success():
    client.post("/api/auth/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "test123"
    })

    response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "test123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password():
    client.post("/api/auth/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "test123"
    })

    response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "wrong"
    })
    assert response.status_code == 401


def test_login_nonexistent_user():
    response = client.post("/api/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "test123"
    })
    assert response.status_code == 401


def test_get_current_user_without_token():
    response = client.get("/api/users/me")
    assert response.status_code == 403


def test_get_current_user_with_valid_token():
    signup_response = client.post("/api/auth/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "test123"
    })
    user_data = signup_response.json()

    login_response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "test123"
    })
    token_data = login_response.json()

    response = client.get("/api/users/me", headers={
        "Authorization": f"Bearer {token_data['access_token']}"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_data["id"]
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "created_at" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_get_current_user_with_invalid_token():
    response = client.get("/api/users/me", headers={
        "Authorization": "Bearer invalidtoken"
    })
    assert response.status_code == 401


def test_user_response_contract_compliance():
    response = client.post("/api/auth/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "test123"
    })
    data = response.json()

    required_fields = ["id", "name", "email", "created_at"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    forbidden_fields = ["password", "password_hash", "token"]
    for field in forbidden_fields:
        assert field not in data, f"Forbidden field present in response: {field}"
