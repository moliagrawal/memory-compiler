"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from mc.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "gpu" in data
    assert data["vram_limit_safe"] is True


def test_list_videos(client):
    res = client.get("/api/videos")
    assert res.status_code == 200
    data = res.json()
    assert "videos" in data
    assert len(data["videos"]) > 0


def test_gallery_endpoint(client):
    res = client.get("/api/gallery")
    assert res.status_code == 200
    data = res.json()
    assert "cases" in data
    assert len(data["cases"]) == 3


def test_search_endpoint(client):
    res = client.post("/api/search", json={
        "query": "person opens the refrigerator door",
        "video_id": "test_video_001",
        "model_mode": "COMPARE",
        "top_k": 3
    })
    assert res.status_code == 200
    data = res.json()
    assert "primary_predictions" in data
    assert len(data["primary_predictions"]) > 0
    assert "comparison" in data
    assert "timeline_heatmap" in data
