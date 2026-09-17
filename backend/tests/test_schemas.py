import pytest
from pydantic import ValidationError
from app.schemas.search import SearchRequest


def test_search_request_valid():
    req = SearchRequest(query="person wearing yellow jacket", top_k=5)
    assert req.query == "person wearing yellow jacket"
    assert req.top_k == 5


def test_search_request_query_too_short():
    with pytest.raises(ValidationError):
        SearchRequest(query="a")


def test_search_request_top_k_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="hello world", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="hello world", top_k=101)
