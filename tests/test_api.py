"""Тесты API. Ollama и torch не нужны: SupportSystem подменяется фейком."""
import pytest
from fastapi.testclient import TestClient

import api
import prompts


class FakeSystem:
    """Заглушка SupportSystem: записывает аргументы, возвращает предсказуемый ответ."""

    def __init__(self, exc: Exception | None = None):
        self.calls = []
        self.exc = exc

    def process_query(self, user_id, mode, client_question, account_info):
        self.calls.append((user_id, mode, client_question, account_info))
        if self.exc:
            raise self.exc
        return "поисковый запрос", "ответ клиенту", "заметка оператору"


@pytest.fixture(autouse=True)
def reset_system():
    """Глобальная ссылка на систему не должна протекать между тестами."""
    api.system = None
    yield
    api.system = None


@pytest.fixture
def client():
    # Без `with` контекстный менеджер не запускает lifespan,
    # поэтому настоящая (тяжелая) SupportSystem не создается.
    return TestClient(api.app)


@pytest.fixture
def fake_system():
    fake = FakeSystem()
    api.system = fake
    return fake


def test_health_reports_system_not_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "system_loaded": False}


def test_chat_returns_503_before_startup(client):
    response = client.post("/chat", json={"client_question": "Почему списали деньги?"})

    assert response.status_code == 503


def test_chat_forwards_all_arguments_to_pipeline(client, fake_system):
    response = client.post(
        "/chat",
        json={
            "client_question": "Почему арест?",
            "user_id": "client-42",
            "mode": "pizza_staff",
            "account_info": "Долг 10 000 ₽",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "search_query": "поисковый запрос",
        "client_answer": "ответ клиенту",
        "operator_note": "заметка оператору",
    }
    # Главная проверка: аргументы доходят в правильном порядке
    # (именно здесь раньше был TypeError -> 500).
    assert fake_system.calls == [
        ("client-42", "pizza_staff", "Почему арест?", "Долг 10 000 ₽")
    ]


def test_chat_uses_defaults_for_optional_fields(client, fake_system):
    response = client.post("/chat", json={"client_question": "Вопрос"})

    assert response.status_code == 200
    assert fake_system.calls == [("anonymous", "bank_support", "Вопрос", "Нет данных")]


def test_chat_rejects_unknown_mode(client, fake_system):
    response = client.post(
        "/chat", json={"client_question": "Вопрос", "mode": "нет_такого_режима"}
    )

    assert response.status_code == 422
    assert fake_system.calls == []


def test_chat_rejects_empty_question(client, fake_system):
    response = client.post("/chat", json={"client_question": ""})

    assert response.status_code == 422


def test_chat_hides_internal_errors(client, fake_system):
    api.system = FakeSystem(exc=RuntimeError("Ollama недоступна"))

    response = client.post("/chat", json={"client_question": "Вопрос"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Внутренняя ошибка сервера"}


def test_modes_endpoint_lists_all_modes(client):
    response = client.get("/modes")

    assert response.status_code == 200
    assert response.json() == {key: data["name"] for key, data in prompts.MODES.items()}


def test_lifespan_loads_and_clears_system(monkeypatch):
    loaded = FakeSystem()
    monkeypatch.setattr(api, "load_system", lambda: loaded)

    with TestClient(api.app) as client:
        assert client.get("/health").json()["system_loaded"] is True
        assert api.system is loaded

    assert api.system is None
