"""
FastAPI приложение: генератор блог-постов с контекстом из NewsAPI и генерацией через OpenAI.

Что делает сервис:
1) Принимает тему поста (topic).
2) Получает свежие новости из NewsAPI.org по этой теме.
3) Использует новости как контекст и генерирует:
   - привлекательный заголовок,
   - meta description,
   - полный текст статьи (структурированный, с подзаголовками).
4) Имеет эндпоинты для проверки работоспособности (health/heartbeat/ready).
5) Запускается через uvicorn.

Установка зависимостей (пример):
pip install fastapi uvicorn requests pydantic-settings openai

Переменные окружения:
- OPENAI_API_KEY
- NEWSAPI_KEY
- (опционально) OPENAI_MODEL, PORT, NEWS_LANGUAGE, NEWS_PAGE_SIZE, NEWS_DAYS_BACK, REQUEST_TIMEOUT_SEC, OPENAI_MAX_OUTPUT_TOKENS
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

# Для Pydantic v2: pydantic-settings вынесен в отдельный пакет.
from pydantic_settings import BaseSettings, SettingsConfigDict

from openai import OpenAI


# ----------------------------
# Настройки приложения
# ----------------------------

class Settings(BaseSettings):
    """
    Настройки приложения читаются из переменных окружения.

    model_config позволяет:
    - автоматически подхватывать env переменные,
    - (опционально) читать .env файл при локальной разработке.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Обязательные ключи:
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    newsapi_key: str = Field(..., alias="NEWSAPI_KEY")

    # OpenAI:
    openai_model: str = Field("gpt-4.1-nano", alias="OPENAI_MODEL")
    openai_temperature: float = Field(0.5, alias="OPENAI_TEMPERATURE")
    openai_max_output_tokens: int = Field(2000, alias="OPENAI_MAX_OUTPUT_TOKENS")

    # NewsAPI:
    newsapi_base_url: str = Field("https://newsapi.org/v2/everything", alias="NEWSAPI_BASE_URL")
    news_language: str = Field("en", alias="NEWS_LANGUAGE")
    news_page_size: int = Field(5, alias="NEWS_PAGE_SIZE")
    news_days_back: int = Field(7, alias="NEWS_DAYS_BACK")

    # Сеть/таймауты:
    request_timeout_sec: int = Field(15, alias="REQUEST_TIMEOUT_SEC")

    # Сервер:
    port: int = Field(8000, alias="PORT")


def load_settings() -> Settings:
    """
    Загружает настройки и в случае ошибок в env — поднимает понятное исключение.
    """
    try:
        return Settings()
    except ValidationError as e:
        # Понятное сообщение о том, какие env переменные не заданы.
        raise RuntimeError(
            "Ошибка конфигурации: убедитесь, что заданы переменные окружения "
            "OPENAI_API_KEY и NEWSAPI_KEY.\n"
            f"Details: {e}"
        ) from e


settings = None
settings_error: Optional[str] = None

try:
    settings = load_settings()
except Exception as e:
    # Не “роняем” импорт модуля: приложение сможет стартовать,
    # но эндпоинт /ready будет показывать проблему.
    settings_error = str(e)

# Глобальная HTTP-сессия, чтобы переиспользовать соединения (быстрее и экономнее).
http_session = requests.Session()


# ----------------------------
# Pydantic-модели запросов/ответов
# ----------------------------

class TopicRequest(BaseModel):
    """
    Вход: тема поста + опциональные параметры для управления поиском новостей.
    """
    topic: str = Field(..., min_length=2, max_length=200, description="Тема поста (ключевые слова).")
    language: Optional[str] = Field(None, description="Язык новостей (ISO-639-1), например: en, ru.")
    max_articles: Optional[int] = Field(None, ge=1, le=20, description="Сколько статей использовать как контекст.")
    days_back: Optional[int] = Field(None, ge=1, le=30, description="За сколько дней искать новости.")


class NewsArticle(BaseModel):
    """
    Нормализованное представление статьи из NewsAPI.
    """
    title: str
    source: str
    published_at: str
    url: str
    description: Optional[str] = None


class GeneratedPost(BaseModel):
    """
    Ответ генерации.
    """
    title: str
    meta_description: str
    post_content: str
    used_news_count: int
    news: List[NewsArticle]


# ----------------------------
# Вспомогательные функции: NewsAPI
# ----------------------------

def _iso_date_days_back(days_back: int) -> str:
    """
    Возвращает дату (UTC) в ISO 8601 для параметра `from` в NewsAPI.
    """
    dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    # NewsAPI принимает и дату, и дату-время; используем дату-время для точности.
    return dt.replace(microsecond=0).isoformat()


def fetch_news(topic: str, language: str, page_size: int, days_back: int) -> List[NewsArticle]:
    """
    Получает новости из NewsAPI по endpoint /v2/everything.

    Важно:
    - NewsAPI возвращает "status": "ok" или "error".
    - Ошибки (401/429 и др.) обрабатываем и выдаём понятное сообщение.
    """
    if settings is None:
        # Если конфигурация не загрузилась — сообщаем.
        raise RuntimeError(settings_error or "Settings not loaded.")

    # Защита от некорректных значений page_size.
    page_size = max(1, min(page_size, 100))  # по документации NewsAPI максимум 100

    params = {
        "q": topic,
        "language": language,
        "sortBy": "publishedAt",         # свежие сначала
        "pageSize": page_size,
        "from": _iso_date_days_back(days_back),
    }

    headers = {
        # Рекомендуется передавать ключ в заголовке.
        "X-Api-Key": settings.newsapi_key,
    }

    try:
        resp = http_session.get(
            settings.newsapi_base_url,
            params=params,
            headers=headers,
            timeout=settings.request_timeout_sec,
        )
    except requests.exceptions.Timeout as e:
        raise HTTPException(status_code=504, detail="NewsAPI: таймаут запроса.") from e
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"NewsAPI: ошибка сети: {str(e)}") from e

    # NewsAPI часто возвращает JSON даже при ошибках; пробуем распарсить.
    try:
        payload = resp.json()
    except ValueError:
        payload = None

    if resp.status_code != 200:
        # Пытаемся вытащить нормальный текст ошибки.
        if isinstance(payload, dict) and payload.get("message"):
            message = payload.get("message")
        else:
            message = resp.text[:500]
        raise HTTPException(
            status_code=502,
            detail=f"NewsAPI: HTTP {resp.status_code}: {message}",
        )

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="NewsAPI: неожиданный формат ответа (не JSON-объект).")

    if payload.get("status") != "ok":
        code = payload.get("code", "unknown_error")
        message = payload.get("message", "Unknown error from NewsAPI.")
        raise HTTPException(status_code=502, detail=f"NewsAPI: {code}: {message}")

    articles = payload.get("articles", []) or []
    normalized: List[NewsArticle] = []

    for a in articles:
        # Подстраховка от неполных данных.
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        published_at = (a.get("publishedAt") or "").strip()
        description = (a.get("description") or None)
        source_name = ""
        src = a.get("source") or {}
        if isinstance(src, dict):
            source_name = (src.get("name") or "").strip()

        if not title or not url:
            continue

        normalized.append(
            NewsArticle(
                title=title,
                source=source_name or "Unknown",
                published_at=published_at or "",
                url=url,
                description=description,
            )
        )

    return normalized


def render_news_context(news: List[NewsArticle]) -> str:
    """
    Превращает список новостей в компактный контекст для промпта.
    """
    if not news:
        return "Нет найденных новостей по теме (контекст отсутствует)."

    lines = []
    for i, n in enumerate(news, start=1):
        # В контекст полезно включать ссылку и краткое описание.
        desc = (n.description or "").strip()
        if desc:
            desc = desc.replace("\n", " ").strip()
        lines.append(
            f"{i}) {n.title}\n"
            f"   Source: {n.source}\n"
            f"   PublishedAt: {n.published_at}\n"
            f"   URL: {n.url}\n"
            f"   Snippet: {desc if desc else '—'}"
        )
    return "\n".join(lines)


# ----------------------------
# Вспомогательные функции: OpenAI
# ----------------------------

def extract_output_text(response_obj: Any) -> str:
    """
    Надёжно извлекает текст из Responses API, не полагаясь на response.output_text
    (в некоторых версиях SDK он может отсутствовать).

    Структура: response.output -> items -> item.content -> content blocks -> text.
    """
    chunks: List[str] = []
    output = getattr(response_obj, "output", None)

    if not output:
        return ""

    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)

    return "".join(chunks).strip()


def generate_post_with_openai(topic: str, news_context: str, temperature: float, max_output_tokens: int) -> Dict[str, Any]:
    """
    Генерирует JSON с полями title/meta_description/post_content через Responses API.

    Чтобы получить стабильную структуру, используем JSON Schema (Structured Outputs).
    """
    if settings is None:
        raise RuntimeError(settings_error or "Settings not loaded.")

    client = OpenAI(api_key=settings.openai_api_key)

    # JSON Schema: строгий формат выходных данных.
    # Важно: additionalProperties=False помогает “зажать” модель в рамки.
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 5},
            "meta_description": {"type": "string", "minLength": 50},
            "post_content": {"type": "string", "minLength": 800},
        },
        "required": ["title", "meta_description", "post_content"],
        "additionalProperties": False,
    }

    # Промпт: новости — это контекст, который должен быть использован (где уместно).
    prompt = f"""
Ты — редактор и автор блога. Сгенерируй материал на тему: "{topic}".

Ниже — актуальные новости (контекст). Используй их как примеры и опорные факты, где это уместно:
{news_context}

Требования:
1) Заголовок: привлекательный, точный, без кликбейта.
2) Meta description: 150–300 символов, информативно, с ключевыми словами.
3) Текст статьи:
   - Не менее 1500 символов (лучше больше).
   - Чёткая структура: вступление, основная часть, заключение.
   - В основной части: подзаголовки, анализ текущих трендов, упоминание 2–5 новостей из контекста.
   - Каждый абзац: минимум 3–4 предложения.
   - Пиши понятно, без воды.

Верни результат строго в JSON по схеме.
""".strip()

    try:
        # Responses API: используем max_output_tokens для ограничения длины генерации.
        resp = client.responses.create(
            model=settings.openai_model,
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            store=False,  # не сохранять запрос/ответ на стороне API, если это допустимо вашей политикой
            text={
                "format": {
                    "type": "json_schema",
                    "name": "blog_post",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI: ошибка генерации: {str(e)}") from e

    raw_text = extract_output_text(resp)
    if not raw_text:
        raise HTTPException(status_code=502, detail="OpenAI: пустой ответ модели.")

    # Пытаемся распарсить JSON.
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Иногда модель может вернуть обёртки; дадим более полезную диагностику.
        raise HTTPException(
            status_code=502,
            detail="OpenAI: модель вернула невалидный JSON. "
                   "Попробуйте уменьшить сложность темы или увеличить max_output_tokens."
        )

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="OpenAI: неожиданный формат (не JSON-объект).")

    return data


# ----------------------------
# FastAPI приложение и эндпоинты
# ----------------------------

app = FastAPI(
    title="Blog Post Generator API",
    version="2.0.0",
    description="Генерация блог-постов с контекстом из NewsAPI + OpenAI.",
)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Простой эндпоинт, чтобы проверить что сервис запущен.
    """
    return {"message": "Service is running"}


@app.get("/heartbeat")
async def heartbeat() -> Dict[str, str]:
    """
    Heartbeat — быстрый ответ без внешних зависимостей.
    """
    return {"status": "OK"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health — показывает базовый статус и наличие конфигурации.
    Внешние API не дергаем, чтобы не тратить квоты.
    """
    ok = settings is not None
    return {
        "status": "OK" if ok else "DEGRADED",
        "config_loaded": ok,
        "error": None if ok else (settings_error or "Unknown settings error"),
    }


@app.get("/ready")
async def readiness() -> Dict[str, Any]:
    """
    Readiness — “готов ли сервис обслуживать генерацию”.
    Проверяем наличие ключей и минимальную возможность вызвать внешние зависимости.
    (Можно расширять: реально сделать тестовый запрос к NewsAPI/OpenAI.)
    """
    if settings is None:
        raise HTTPException(status_code=503, detail=f"Not ready: {settings_error}")

    # Минимальная проверка: ключи не пустые.
    if not settings.openai_api_key.strip():
        raise HTTPException(status_code=503, detail="Not ready: OPENAI_API_KEY is empty.")
    if not settings.newsapi_key.strip():
        raise HTTPException(status_code=503, detail="Not ready: NEWSAPI_KEY is empty.")

    return {"status": "READY"}


@app.post("/generate-post", response_model=GeneratedPost)
async def generate_post(req: TopicRequest) -> GeneratedPost:
    """
    Основной эндпоинт генерации.

    Алгоритм:
    1) Получаем новости из NewsAPI.
    2) Формируем контекст.
    3) Генерируем результат в OpenAI (title/meta/article) в строгом JSON.
    4) Возвращаем клиенту.
    """
    if settings is None:
        raise HTTPException(status_code=503, detail=f"Service not configured: {settings_error}")

    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=422, detail="Поле topic не должно быть пустым.")

    language = (req.language or settings.news_language).strip()
    days_back = req.days_back or settings.news_days_back
    page_size = req.max_articles or settings.news_page_size

    # 1) Получаем новости.
    news = fetch_news(
        topic=topic,
        language=language,
        page_size=page_size,
        days_back=days_back,
    )

    # 2) Формируем контекст.
    news_context = render_news_context(news)

    # 3) Генерация.
    data = generate_post_with_openai(
        topic=topic,
        news_context=news_context,
        temperature=settings.openai_temperature,
        max_output_tokens=settings.openai_max_output_tokens,
    )

    # 4) Собираем итоговый объект ответа.
    # Подстраховка: если модель вернула строки с пробелами — чистим.
    title = (data.get("title") or "").strip()
    meta_description = (data.get("meta_description") or "").strip()
    post_content = (data.get("post_content") or "").strip()

    if not title or not meta_description or not post_content:
        raise HTTPException(status_code=502, detail="OpenAI: отсутствуют обязательные поля в ответе.")

    return GeneratedPost(
        title=title,
        meta_description=meta_description,
        post_content=post_content,
        used_news_count=len(news),
        news=news,
    )


# ----------------------------
# Точка входа (uvicorn)
# ----------------------------

if __name__ == "__main__":
    import uvicorn

    # Если PORT не задан, используется значение из Settings (по умолчанию 8000).
    port = 8000
    try:
        port = int(os.getenv("PORT", str(settings.port if settings else 8000)))
    except ValueError:
        port = 8000

    uvicorn.run("app:app", host="0.0.0.0", port=port)
