# src/utils_external.py

import wikipedia
import requests

def get_wikipedia_answer(query, lang="en"):
    """
    Search Wikipedia for a summary on the query in the given language.
    """
    try:
        wikipedia.set_lang(lang)
        results = wikipedia.search(query)
        if not results:
            return None, None
        page = wikipedia.page(results[0])
        return page.summary, page.url
    except Exception:
        return None, None

def get_duckduckgo_answer(query):
    """
    Query DuckDuckGo Instant Answer API for a snippet answer.
    """
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1})
        data = resp.json()
        if data.get("AbstractText"):
            return data["AbstractText"], data.get("AbstractURL")
        elif data.get("Answer"):
            return data["Answer"], data.get("AbstractURL")
        elif data.get("RelatedTopics"):
            topic = data["RelatedTopics"][0]
            if isinstance(topic, dict) and "Text" in topic:
                return topic["Text"], topic.get("FirstURL")
        return None, None
    except Exception:
        return None, None