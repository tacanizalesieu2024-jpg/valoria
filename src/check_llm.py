"""Check the LLM connection.   Run:  python -m src.check_llm
Or test specific models:   python -m src.check_llm gemini-3.8-flash gemini-flash-latest
Sends one tiny request to each model and prints the result. Never prints your key."""
import sys
import time

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_FALLBACK_MODELS, LLM_MODEL


def main():
    if not LLM_API_KEY:
        print("No key found: add LLM_API_KEY=... to the .env file")
        return
    print(f"Key found ({len(LLM_API_KEY)} characters, starts with '{LLM_API_KEY[:3]}')")
    print(f"Endpoint: {LLM_BASE_URL or 'OpenAI default'}\n")
    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None, max_retries=0, timeout=60)
    if not sys.argv[1:]:  # no models given: also show which ones the key can use
        try:
            names = sorted(m.id.replace("models/", "") for m in client.models.list())
            print("Models available to your key:", ", ".join(n for n in names if "gemini" in n))
            print()
        except Exception as e:  # noqa: BLE001
            print(f"Could not list models: {type(e).__name__}\n")
    models = sys.argv[1:] or [LLM_MODEL] + LLM_FALLBACK_MODELS
    for model in dict.fromkeys(models):
        start = time.time()
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with the word ok"}])
            print(f"OK    {model}: {r.choices[0].message.content.strip()[:40]}  ({time.time() - start:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {model}: {type(e).__name__}: {str(e).replace(LLM_API_KEY, '[key]')[:300]}")


if __name__ == "__main__":
    main()
