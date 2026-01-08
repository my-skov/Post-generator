"""
FastAPI приложение: генератор блог-постов на основе актуальных новостей.

Возможности:
- Получение свежих новостей по теме через Currents News API (latest-news endpoint).
- Использование новостей как контекста при генерации контента через OpenAI Chat Completions.
- Настройки через переменные окружения (API ключи, таймауты, параметры генерации).
- Эндпоинты: /, /heartbeat, /health, /generate-post.
- Запуск через uvicorn.

Документация:
- Currents latest-news endpoint: https://api.currentsapi.services/v1/latest-news
- OpenAI модель GPT-4.1 nano (alias: gpt-4.1-nano)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Вариант SDK OpenAI "старого" стиля (как в вашем Colab-коде):
# openai.ChatCompletion.create(...)
import openai


# -------------------------
# Настройки через окружение
# -------------------------

def _get_env(name: str, default: Optional[str] = None) -> str:
    """
    Безопасное чтение переменной окружения.
    Если default не задан и переменной нет — выбрасываем понятную ошибку.
    """
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"Переменная окружения {name} должна быть установлена")
    return value


# Обязательные ключи (приложение не стартует без них)
OPENAI_API_KEY = _get_env("OPENAI_API_KEY")
CURRENTS_API_KEY = _get_env("CURRENTS_API_KEY")

# Опциональные настройки (есть значения по умолчанию)
CURRENTS_BASE_URL = os.getenv("CURRENTS_BASE_URL", "https://api.currentsapi.services/v1/latest-news")
CURRENTS_LANGUAGE = os.getenv("CURRENTS_LANGUAGE", "en")
CURRENTS_TIMEOUT_SEC = float(os.getenv("CURRENTS_TIMEOUT_SEC", "15"))
CURRENTS_MAX_ARTICLES = int(os.getenv("CURRENTS_MAX_ARTICLES", "5"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")  # требование: gpt-4.1-nano
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.5"))
GEN_PRESENCE_PENALTY = float(os.getenv("GEN_PRESENCE_PENALTY", "0.6"))
GEN_FREQUENCY_PENALTY = float(os.getenv("GEN_FREQUENCY_PENALTY", "0.6"))

# Настраиваем OpenAI SDK ключом
openai.api_key = OPENAI_API_KEY


# -------------------------
# FastAPI приложение
# -------------------------

app = FastAPI(
    title="Blog Post Generator (Currents + OpenAI)",
    version="1.0.0",
)


# -------------------------
# Pydantic модели запросов/ответов
# -------------------------

class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120, description="Тема статьи/поста")


class GeneratePostResponse(BaseModel):
    title: str
    meta_description: str
    post_content: str
    news_titles: List[str] = Field(default_factory=list)
    news_context: str


# -------------------------
# Интеграция с Currents API
# -------------------------

def get_recent_news(topic: str) -> Dict[str, Any]:
    """
    Получает свежие новости по теме через Currents API (latest-news endpoint). [page:1]

    ВАЖНО:
    Документация endpoint указывает параметр language (по умолчанию en). [page:1]
    В исходном Colab-коде использовался параметр keywords — он сохранён для совместимости,
    т.к. на практике многие примеры используют фильтрацию по ключевым словам.
    Если Currents вернёт ошибку, она будет проброшена как HTTPException.
    """
    params = {
        "language": CURRENTS_LANGUAGE,   # фильтр языка (документирован) [page:1]
        "keywords": topic,              # фильтр по теме (как в вашем исходнике)
        "apiKey": CURRENTS_API_KEY,     # ключ доступа
    }

    try:
        resp = requests.get(CURRENTS_BASE_URL, params=params, timeout=CURRENTS_TIMEOUT_SEC)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Currents API недоступен: {str(e)}")

    if resp.status_code != 200:
        # Currents в случае ошибок обычно возвращает JSON/text с деталями — отдаём как есть
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка Currents API (HTTP {resp.status_code}): {resp.text}",
        )

    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Currents API вернул не-JSON ответ")

    return data


def build_news_context(currents_payload: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    """
    Собирает удобный контекст из Currents-ответа:
    - список заголовков
    - строка-контекст для промпта
    """
    news_items = currents_payload.get("news", []) or []
    if not isinstance(news_items, list):
        news_items = []

    # Берём первые N статей
    selected = news_items[: max(0, limit)]

    titles: List[str] = []
    context_lines: List[str] = []

    for i, item in enumerate(selected, start=1):
        title = (item or {}).get("title") or ""
        desc = (item or {}).get("description") or ""
        published = (item or {}).get("published") or ""
        source = (item or {}).get("url") or ""

        if title.strip():
            titles.append(title.strip())

        # Делает контекст более полезным, чем просто список заголовков
        line_parts = []
        if title:
            line_parts.append(f"Title: {title}")
        if desc:
            line_parts.append(f"Description: {desc}")
        if published:
            line_parts.append(f"Published: {published}")
        if source:
            line_parts.append(f"URL: {source}")

        if line_parts:
            context_lines.append(f"{i}) " + " | ".join(line_parts))

    if not context_lines:
        news_context = "Свежих релевантных новостей по теме не найдено."
    else:
        news_context = "Актуальные новости по теме:\n" + "\n".join(context_lines)

    return {"titles": titles, "news_context": news_context}


# -------------------------
# Интеграция с OpenAI
# -------------------------

def _openai_chat(messages: List[Dict[str, str]], max_tokens: int) -> str:
    """
    Обёртка над OpenAI ChatCompletion для единообразной обработки ошибок.
    Использует модель gpt-4.1-nano (alias) по умолчанию. [page:2]
    """
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=GEN_TEMPERATURE,
            presence_penalty=GEN_PRESENCE_PENALTY,
            frequency_penalty=GEN_FREQUENCY_PENALTY,
            request_timeout=OPENAI_TIMEOUT_SEC,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # В проде стоит логировать e с traceback; здесь возвращаем понятную ошибку клиенту
        raise HTTPException(status_code=502, detail=f"Ошибка OpenAI API: {str(e)}")


def generate_content(topic: str) -> GeneratePostResponse:
    """
    Генерация контента:
    1) Забираем новости
    2) Строим контекст
    3) Генерируем title, meta, post
    """
    currents_payload = get_recent_news(topic)
    news_pack = build_news_context(currents_payload, limit=CURRENTS_MAX_ARTICLES)

    news_context = news_pack["news_context"]
    news_titles = news_pack["titles"]

    # 1) Заголовок
    title = _openai_chat(
        messages=[
            {
                "role": "system",
                "content": "Ты редактор и SEO-копирайтер. Пиши точно, без кликбейта, но интересно.",
            },
            {
                "role": "user",
                "content": (
                    f"Тема: {topic}\n\n"
                    f"{news_context}\n\n"
                    "Придумай 1 привлекательный и точный заголовок статьи на русском языке. "
                    "Заголовок должен отражать суть темы и учитывать контекст новостей."
                ),
            },
        ],
        max_tokens=80,
    )

    # 2) Мета-описание (без точки-стопа, чтобы не обрезать по первой точке)
    meta_description = _openai_chat(
        messages=[
            {
                "role": "system",
                "content": "Ты SEO-специалист. Пиши мета-описания ясно и информативно.",
            },
            {
                "role": "user",
                "content": (
                    f"Заголовок: {title}\n"
                    f"Тема: {topic}\n\n"
                    f"{news_context}\n\n"
                    "Напиши meta description на русском (примерно 140–170 символов), "
                    "с ключевыми словами и без лишней воды."
                ),
            },
        ],
        max_tokens=160,
    )

    # 3) Полная статья
    post_content = _openai_chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты опытный журналист и аналитик. Пиши структурно, с подзаголовками, "
                    "в деловом и понятном стиле."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Тема статьи: {topic}\n\n"
                    f"{news_context}\n\n"
                    "Требования к статье:\n"
                    "- Русский язык.\n"
                    "- Не менее 1500 символов.\n"
                    "- Чёткая структура: вступление, 2–4 подзаголовка, заключение.\n"
                    "- Упомяни и аккуратно интерпретируй факты/сигналы из новостей (без выдуманных деталей).\n"
                    "- Добавь анализ трендов и практические выводы.\n"
                    "- Каждый абзац: 3–4 предложения.\n"
                ),
            },
        ],
        max_tokens=1500,
    )

    return GeneratePostResponse(
        title=title,
        meta_description=meta_description,
        post_content=post_content,
        news_titles=news_titles,
        news_context=news_context,
    )


# -------------------------
# Эндпоинты
# -------------------------

@app.get("/")
async def root() -> Dict[str, str]:
    """Простой эндпоинт для проверки, что сервис запущен."""
    return {"message": "Service is running"}


@app.get("/heartbeat")
async def heartbeat() -> Dict[str, str]:
    """Эндпоинт для liveness-проверок (например, Kubernetes livenessProbe)."""
    return {"status": "OK"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Эндпоинт для более детальной проверки состояния:
    - показывает, что приложение живо
    - показывает базовые настройки без раскрытия секретов
    """
    return {
        "status": "OK",
        "time": int(time.time()),
        "currents": {
            "base_url": CURRENTS_BASE_URL,
            "language": CURRENTS_LANGUAGE,
            "max_articles": CURRENTS_MAX_ARTICLES,
            "timeout_sec": CURRENTS_TIMEOUT_SEC,
        },
        "openai": {
            "model": OPENAI_MODEL,
            "timeout_sec": OPENAI_TIMEOUT_SEC,
        },
    }


@app.post("/generate-post", response_model=GeneratePostResponse)
async def generate_post_api(payload: TopicRequest) -> GeneratePostResponse:
    """
    Основной эндпоинт генерации:
    Принимает тему, возвращает заголовок, meta-description, статью и контекст новостей.
    """
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=422, detail="Поле topic не должно быть пустым")

    return generate_content(topic)


# -------------------------
# Запуск через uvicorn
# -------------------------

if __name__ == "__main__":
    import uvicorn

    # Порт по умолчанию 8000, можно переопределить через PORT
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # в разработке можно поставить True
    )
