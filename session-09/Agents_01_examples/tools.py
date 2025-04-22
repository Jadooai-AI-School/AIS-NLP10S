import requests
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from agents import function_tool

######################################################################################################
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")

@function_tool
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetches and returns the current temperature for the specified coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error fetching weather data: {e}"

    try:
        times = data['hourly']['time']
        temperatures = data['hourly']['temperature_2m']
        current_time = datetime.now(timezone.utc)

        # Find the closest time index
        time_objects = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in times]
        closest_index = min(range(len(time_objects)), key=lambda i: abs(time_objects[i] - current_time))
        closest_time = time_objects[closest_index]
        current_temp = temperatures[closest_index]

        return (
            f"🌡️ Current Temperature Report:\n"
            f"Location: ({latitude}, {longitude})\n"
            f"Time: {closest_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Temperature: {current_temp}°C"
        )
    except (KeyError, ValueError, IndexError) as e:
        return f"Error processing weather data: {e}"

######################################################################################################
import wikipedia

@function_tool
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia and returns summaries of the top relevant pages."""
    try:
        page_titles = wikipedia.search(query)
        if not page_titles:
            return "🔍 No relevant Wikipedia articles found."

        summaries = []
        for title in page_titles[:3]:
            try:
                summary = wikipedia.summary(title, sentences=2)
                summaries.append(f"📄 **{title}**\n{summary}")
            except wikipedia.exceptions.DisambiguationError:
                summaries.append(f"⚠️ **{title}** has multiple meanings. Please specify further.")
            except wikipedia.exceptions.PageError:
                summaries.append(f"❌ **{title}** page not found.")

        return "\n\n".join(summaries)
    except Exception as e:
        return f"An error occurred while searching Wikipedia: {e}"
######################################################################################################
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

class DuckDuckGoNewsInput(BaseModel):
    query: str = Field(..., description="Search query for DuckDuckGo News")

@function_tool
def duckduckgo_news_search(query: str) -> str:
    """Perform a news search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query)
            news_items = list(results)
            if not news_items:
                return "No news articles found for your query."

            top_articles = news_items[:3]
            formatted_results = []
            for article in top_articles:
                title = article.get('title', 'No title')
                link = article.get('url', 'No URL')
                snippet = article.get('body', 'No summary')
                formatted_results.append(f"📰 {title}\n🔗 {link}\n📝 {snippet}")

            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"An error occurred during the news search: {e}"
######################################################################################################
import os
class SerperSearchInput(BaseModel):
    query: str = Field(..., description="Search query for Serper")

@function_tool
def serper_search(query: str) -> str:
    """Perform a web search using the Serper API."""
    SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Replace with your actual API key
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    data = {"q": query}
    response = requests.post("https://google.serper.dev/search", headers=headers, json=data)
    if response.status_code == 200:
        results = response.json()
        snippets = [item["snippet"] for item in results.get("organic", [])[:3]]
        return "\n\n".join(snippets)
    else:
        raise Exception(f"Serper API request failed with status code {response.status_code}")
