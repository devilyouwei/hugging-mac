"""Full Vela public-contract cases; routes follow the pinned upstream configuration."""

import json
from pathlib import Path
from typing import Any


def cases(source: Path, decision: Any) -> list[dict[str, Any]]:
    def row(key: str, state: Any, questions: Any) -> dict[str, Any]:
        return {"id": key, "state": state, "questions": questions}

    question = {
        "type": "choice",
        "instructions": "What is the request about?",
        "criteria": {
            "coding": "Software development",
            "cooking": "Food preparation",
            "billing": "Payments",
        },
    }
    result = [
        row(
            "decision-multilingual",
            "Ma carte a été débitée deux fois, je veux être remboursée.",
            {"topic": question},
        ),
        row(
            "decision-structured-multiple",
            {"request": "Please explain Python lists", "priority": 2},
            {
                "topic": question,
                "urgent": {
                    "type": "noul",
                    "instructions": "Is urgent action needed?",
                    "criteria": {"true": "An urgent deadline", "false": "No urgency"},
                },
            },
        ),
        row(
            "decision-single-option",
            "Python programming",
            {"topic": {**question, "criteria": {"coding": "Software"}}},
        ),
        row("empty-questions", "Hello", {}),
        row("invalid-question", "Hello", {"bad": {"type": "unsupported-kind"}}),
    ]
    for length in (1024, 2048):
        overhead = len(decision.encode("", question)["ids"])
        text = " hello" * (length - overhead)
        actual = len(decision.encode(text, question)["ids"])
        if actual != length:
            raise ValueError(f"Long-input fixture changed: expected {length}, got {actual}")
        result.append(row(f"decision-{length}", text, {"topic": question}))
    router = json.loads((source / "router_questions.json").read_text())["questions"]
    for wanted in ("domain", "factcheck", "feedback"):
        text, spec = next((t, s) for t, s in router.items() if s["id"] == wanted)
        result.append(
            row(
                f"router-{wanted}",
                "Please check whether Paris is the capital of France.",
                {"signal": {"type": spec["type"], "instructions": text}},
            )
        )
    for kind in ("choice", "set"):
        text, spec = next(
            (t, s)
            for t, s in router.items()
            if s["type"] == kind and s["id"] not in {"domain", "factcheck", "feedback"}
        )
        result.append(
            row(
                f"router-{kind}",
                "Hello, please help me plan a family picnic.",
                {"signal": {"type": kind, "instructions": text}},
            )
        )
    result.extend(
        [
            row(
                "span-described",
                "Alice bought apples and oranges.",
                {"fruit": {"type": "span", "criteria": {"fruit": "a fruit"}}},
            ),
            row(
                "span-toxic",
                "A calm and friendly discussion.",
                {"safety": {"type": "span", "criteria": {"Violence": "physical violence"}}},
            ),
            row(
                "ner-names-only",
                "Alice works at Acme in Paris.",
                {"entities": {"type": "span", "criteria": ["person", "company", "city"]}},
            ),
            row(
                "relation",
                "Alice works at Acme in Paris.",
                {
                    "employer": {
                        "type": "span",
                        "instructions": "Which spans are the employer of Alice?",
                        "criteria": {"employer": "the employer organization"},
                    }
                },
            ),
            row(
                "pii",
                "Contact Marie Dubois at marie.dubois@example.com.",
                {
                    "pii": {
                        "type": "span",
                        "criteria": {
                            "PERSON": "a person's name",
                            "EMAIL_ADDRESS": "an e-mail address",
                        },
                    }
                },
            ),
            row(
                "unsupported-claims",
                {
                    "context": "The tower was completed in 1889.",
                    "answer": "The tower was completed in 1899.",
                },
                {
                    "claims": {
                        "type": "span",
                        "criteria": {"unsupported": "a claim not supported by the context"},
                    }
                },
            ),
            row(
                "evidence",
                {
                    "context": "Paris is the capital of France. Berlin is the capital of Germany.",
                    "answer": "Paris is the capital of France.",
                },
                {
                    "support": {
                        "type": "span",
                        "target": "context",
                        "criteria": {"evidence": "text supporting the answer"},
                    }
                },
            ),
            row(
                "mixed-heads",
                "Marie uses Python; email marie@example.com.",
                {
                    "topic": question,
                    "email": {"type": "span", "criteria": {"EMAIL_ADDRESS": "an e-mail address"}},
                },
            ),
        ]
    )
    return result
