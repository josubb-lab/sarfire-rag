# test_tavily.py
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Test simple
response = client.search(
    query="incendios forestales protocolo actuación",
    max_results=3
)

print("✅ Tavily funciona!")
print(f"Resultados: {len(response.get('results', []))}")
for i, result in enumerate(response.get('results', []), 1):
    print(f"{i}. {result['title']}")