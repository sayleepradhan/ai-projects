import pytest
from src.triple_extractor import parse_triples, KG_DELIMITER

class TestParseTriples:
    def test_parses_single_triple(self):
        raw = "(BERT, is a, language model)"
        result = parse_triples(raw)
        assert result == [("BERT", "is a", "language model")]


    def test_parses_multiple_triples(self):
        raw = (f"(A, relates to, B){KG_DELIMITER}"
               f"(C, extends, D)")
        result = parse_triples(raw)
        assert len(result) == 2


    def test_handles_none_output(self):
        assert parse_triples("NONE") == []


    def test_handles_empty_string(self):
        assert parse_triples("") == []


    def test_handles_null_input(self):
        assert parse_triples(None) == []


    def test_strips_whitespace(self):
        raw = "( BERT , is a , model )"
        result = parse_triples(raw)
        assert result == [("BERT", "is a", "model")]


    def test_ignores_malformed_segments(self):
        raw = f"(A, B, C){KG_DELIMITER}not a triple{KG_DELIMITER}(D, E, F)"
        result = parse_triples(raw)
        assert len(result) == 2
        assert result[0] == ("A", "B", "C")
        assert result[1] == ("D", "E", "F")
