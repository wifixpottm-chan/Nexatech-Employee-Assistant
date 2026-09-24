import pytest

from evaluate_rag import CASES
from rag import retrieve_policy_documents


@pytest.mark.parametrize("case", CASES)
def test_policy_retrieval_matches_expected_source(case):
    documents = retrieve_policy_documents(case.question)
    sources = {document.metadata["source"] for document in documents}

    if case.expected_source is None:
        assert not sources
    else:
        assert case.expected_source in sources