"""Run package-only inference and compare with pinned upstream reference probabilities."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from vela_coreai import VelaCoreAI  # type: ignore[import-not-found]


async def main() -> None:
    root = Path(__file__).parent
    cases = json.loads((root / "validation_cases.json").read_text())
    rows = []
    async with VelaCoreAI.open(root) as model:
        for case in cases:
            arrays, labels = model.pack(case["payload"]["state"], case["payload"]["question"])
            for name, expected in case["inputs"].items():
                np.testing.assert_array_equal(arrays[name].ravel(), expected)
            probabilities = await model.probabilities(
                case["payload"]["state"], case["payload"]["question"]
            )
            values = np.array([probabilities[label] for label in labels])
            reference = np.array(case["reference_probabilities"])
            np.testing.assert_allclose(values, reference, atol=0.02, rtol=0.02)
            decision = max(probabilities, key=probabilities.__getitem__)
            assert decision == case["reference_decision"]
            if case.get("expected") is not None:
                assert decision == case["expected"]
            rows.append(
                {
                    "id": case["id"],
                    "sequence_length": int(arrays["attention_mask"].sum()),
                    "option_count": len(labels),
                    "decision": decision,
                    "max_probability_error": float(np.max(np.abs(values - reference))),
                    "probabilities": probabilities,
                }
            )
        first = cases[0]
        arrays, labels = model.pack(first["payload"]["state"], first["payload"]["question"])
        expected = await model.probabilities(
            first["payload"]["state"], first["payload"]["question"]
        )
        fixture = {name: value.ravel().tolist() for name, value in arrays.items()}
        fixture.update(
            expected=[expected[label] for label in labels], labels=labels, output="logits"
        )
        (root / "swift_fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
        rejected = 0
        for state, question in [
            ("text", {"type": "span"}),
            ("text", {"type": "choice", "criteria": {}}),
            ("word " * (model.manifest["sequence_length"] + 1), first["payload"]["question"]),
        ]:
            try:
                model.pack(state, question)
            except ValueError:
                rejected += 1
        assert rejected == 3
        # Host adapter must preserve the public choice/noul response shape.
        answer = await model.predict(first["payload"]["state"], first["payload"]["question"])
        assert answer["type"] == "choice" and answer["choice"] == first["expected"]
        noul = cases[2]
        answer = await model.predict(noul["payload"]["state"], noul["payload"]["question"])
        assert answer["type"] == "noul" and 0 <= answer["noul"] <= 1
        questions = {
            "topic": first["payload"]["question"],
            "software": noul["payload"]["question"],
        }
        answers, usage = await model.predict_questions(first["payload"]["state"], questions)
        assert list(answers) == list(questions) and usage > 0
        for key, question in questions.items():
            assert answers[key] == await model.predict(first["payload"]["state"], question)
        list_question = {"type": "choice", "criteria": ["coding", "cooking", "sports"]}
        map_question = {**list_question, "criteria": dict.fromkeys(list_question["criteria"])}
        left, left_labels = model.pack(None, list_question)
        right, right_labels = model.pack(None, map_question)
        assert left_labels == right_labels
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])
    assert "torch" not in sys.modules
    report = {
        "status": "passed",
        "scope": "dynamic decision graph",
        "device": "cpu",
        "cases": rows,
        "invalid_requests_rejected": rejected,
        "pytorch_imported": False,
        "requires_original_weights": False,
        "multiple_questions": "passed",
        "list_criteria": "passed",
    }
    (root / "package_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
