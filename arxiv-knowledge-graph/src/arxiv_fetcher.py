import requests
import xml.etree.ElementTree as ET

from dataclasses import dataclass

@dataclass
class ArxivPaper:
    """Represents a single ArXiv paper."""
    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def fetch_arxiv_papers(
    query: str,
    max_results: int = 5,
) -> list[ArxivPaper]:
    """Fetch papers from ArXiv matching the query.


    Args:
        query: Search query (e.g., "transformer attention mechanism").
        max_results: Maximum number of papers to return.


    Returns:
        List of ArxivPaper objects.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()
    return _parse_arxiv_response(response.text)


def _parse_arxiv_response(xml_text: str) -> list[ArxivPaper]:
    """Parse ArXiv API XML response into ArxivPaper objects."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        title = " ".join(title.split())  # normalize whitespace

        abstract = entry.find("atom:summary", ns).text.strip()
        abstract = " ".join(abstract.split())

        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
        ]

        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]

        papers.append(ArxivPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            arxiv_id=arxiv_id,
        ))

    return papers

