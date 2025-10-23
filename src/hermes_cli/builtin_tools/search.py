"""Web search tools using SerpAPI."""

import os
import requests
from hermes_cli.tools import tool


@tool(
    name="web_search",
    description="Search the web using Google search and return results including titles, links, and snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to execute"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 10)",
                "default": 5
            }
        },
        "required": ["query"]
    },
    builtin=True
)
def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web using SerpAPI.

    Args:
        query: Search query string
        num_results: Number of results to return (max 10)

    Returns:
        Dict with 'results' list or 'error' key
    """
    # Get API key from environment
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {
            "error": "SERPAPI_API_KEY environment variable not set. "
                    "Get a free API key at https://serpapi.com/ and set it with: "
                    "export SERPAPI_API_KEY='your-api-key'"
        }

    # Limit num_results
    num_results = min(num_results, 10)

    try:
        # Make request to SerpAPI
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "engine": "google"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract organic results
        organic_results = data.get("organic_results", [])

        if not organic_results:
            return {"results": [], "message": "No results found"}

        # Format results
        results = []
        for result in organic_results[:num_results]:
            results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            })

        return {"results": results, "query": query}

    except requests.exceptions.Timeout:
        return {"error": "Search request timed out (30s limit)"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Search request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
