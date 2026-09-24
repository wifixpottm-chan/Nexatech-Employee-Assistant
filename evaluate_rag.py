import json
from dataclasses import asdict, dataclass

from rag import retrieve_policy_documents


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_source: str | None


CASES = (
    EvaluationCase("What is the remote work limit?", "remote_work_policy.md"),
    EvaluationCase("How should I request vacation?", "leave_policy.md"),
    EvaluationCase("How is overtime handled?", "overtime_policy.md"),
    EvaluationCase("What are the core working hours?", "working_hours_policy.md"),
    EvaluationCase("What should I do when I am sick?", "sick_leave_policy.md"),
    EvaluationCase("How do I correct a missing check-in?", "attendance_policy.md"),
    EvaluationCase("Who won the football match yesterday?", None),
)


def evaluate() -> dict:
    results = []
    for case in CASES:
        documents = retrieve_policy_documents(case.question)
        sources = [document.metadata["source"] for document in documents]
        passed = (
            case.expected_source in sources
            if case.expected_source is not None
            else not sources
        )
        results.append(
            {
                **asdict(case),
                "retrieved_sources": sources,
                "passed": passed,
            }
        )

    passed_count = sum(result["passed"] for result in results)
    return {
        "passed": passed_count,
        "total": len(results),
        "accuracy": passed_count / len(results),
        "cases": results,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))