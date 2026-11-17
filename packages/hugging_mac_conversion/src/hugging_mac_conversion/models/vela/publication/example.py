import asyncio
import json
from pathlib import Path

from vela_coreai import VelaCoreAI  # type: ignore[import-not-found]


async def main() -> None:
    async with VelaCoreAI.open(Path(__file__).parent) as model:
        result = await model.predict(
            "请写一个 Python 排序函数。",
            {
                "type": "choice",
                "instructions": "What is the topic?",
                "criteria": {"coding": "Software", "cooking": "Food"},
            },
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            await model.predict(
                "Python code",
                {
                    "type": "noul",
                    "instructions": "Is this about software?",
                },
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
