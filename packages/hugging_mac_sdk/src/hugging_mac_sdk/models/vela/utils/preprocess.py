"""Map validated shared decision questions to the original Vela encoder contract."""

from typing import Any

from hugging_mac_sdk.schemas.decision import DecisionRequest


def prepare_request(request: DecisionRequest) -> dict[str, Any]:
    return {
        "state": request.state,
        "questions": {
            key: question.model_dump(mode="json", exclude_none=True)
            for key, question in request.questions.items()
        },
    }
