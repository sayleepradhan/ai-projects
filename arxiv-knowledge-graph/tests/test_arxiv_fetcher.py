import pytest
import src.arxiv_fetcher



SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>  Test Paper\n  Title  </title>
    <summary>  This is a test abstract.  </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
  </entry>
</feed>"""




class TestParseArxivResponse:
    def test_parses_single_entry(self):
        papers = src.arxiv_fetcher._parse_arxiv_response(SAMPLE_XML)
        assert len(papers) == 1
        assert isinstance(papers[0], src.arxiv_fetcher.ArxivPaper)


    def test_normalizes_whitespace_in_title(self):
        papers = src.arxiv_fetcher._parse_arxiv_response(SAMPLE_XML)
        assert papers[0].title == "Test Paper Title"


    def test_normalizes_whitespace_in_abstract(self):
        papers = src.arxiv_fetcher._parse_arxiv_response(SAMPLE_XML)
        assert papers[0].abstract == "This is a test abstract."


    def test_extracts_authors(self):
        papers = src.arxiv_fetcher._parse_arxiv_response(SAMPLE_XML)
        assert papers[0].authors == ["Jane Doe", "John Smith"]


    def test_extracts_arxiv_id(self):
        papers = src.arxiv_fetcher._parse_arxiv_response(SAMPLE_XML)
        assert papers[0].arxiv_id == "2301.00001v1"

    def test_empty_feed_returns_empty_list(self):
        xml = '''<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>'''
        assert src.arxiv_fetcher._parse_arxiv_response(xml) == []
