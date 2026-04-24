import json
import os
import re
from typing import Dict, Any
from dotenv import load_dotenv

from gigachat import GigaChat
from gigachat.models import (
    Chat,
    Messages,
    MessagesRole,
    Function,
    FunctionParameters,
)

load_dotenv()


def load_data() -> Dict:
    with open("data/components.json", "r", encoding="utf-8") as f:
        return json.load(f)


def is_pc_request(text: str) -> bool:
    keywords = [
        "пк", "pc", "комп", "сборк", "процессор",
        "видеокарт", "игров", "собрать"
    ]
    return any(k in text.lower() for k in keywords)


def extract_params(text: str):
    budget = None
    perf = None

    m = re.search(r"(\d+)\s*\$|\$(\d+)|(\d+)\s*дол", text)
    if m:
        budget = int(next(g for g in m.groups() if g))

    m = re.search(r"коэф.*?(\d+(\.\d+)?)", text)
    if m:
        perf = float(m.group(1))

    return budget, perf


def normalize(text: str):
    budget, perf = extract_params(text.lower())

    if budget is None:
        budget = 800

    if perf is None:
        perf = 3.0 if "игров" in text.lower() else 2.0

    return budget, perf


def build_pc(target_performance: float, budget: float) -> Dict[str, Any]:
    data = load_data()

    cpus = [c for c in data["cpus"] if c["price"] <= budget * 0.4]
    gpus = [g for g in data["gpus"] if g["price"] <= budget * 0.6]

    if not cpus or not gpus:
        return {
            "error": "Нет подходящих комплектующих",
            "available": False
        }

    cpu = min(cpus, key=lambda x: abs(x["performance"] - target_performance))
    gpu = min(gpus, key=lambda x: abs(x["performance"] - target_performance * 1.2))

    return {
        "cpu": cpu,
        "gpu": gpu,
        "total_price": cpu["price"] + gpu["price"],
        "performance": round((cpu["performance"] + gpu["performance"]), 2),
        "budget": budget
    }


AVAILABLE_FUNCTIONS = {
    "build_pc": build_pc,
}


functions_description = [
    Function(
        name="build_pc",
        description="Собирает ПК (CPU + GPU) по бюджету и производительности",
        parameters=FunctionParameters(
            type="object",
            properties={
                "target_performance": {"type": "number"},
                "budget": {"type": "number"},
            },
            required=["target_performance", "budget"],
        ),
    )
]


SYSTEM_PROMPT = """
Ты консультант по сборке ПК.

ВАЖНО:
- Отвечай ТОЛЬКО если запрос про ПК.
- Используй только результат функции build_pc.
- Никогда не выдумывай комплектующие.
- Не задавай вопросы пользователю.

ФОРМАТ:
CPU
GPU
Цена
Производительность
"""


class PCAgent:
    def __init__(self, credentials: str):
        self.client = GigaChat(
            credentials=credentials,
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            temperature=0.3,
            max_tokens=1500,
        )

        self.history = [
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT)
        ]

    def _run_function(self, function_call):
        name = function_call.name

        args = (
            json.loads(function_call.arguments)
            if isinstance(function_call.arguments, str)
            else function_call.arguments
        )

        print(f"\n[FUNCTION] {name} {args}")

        result = AVAILABLE_FUNCTIONS[name](**args)

        print(f"[RESULT] {result}\n")

        return result

    def ask(self, text: str):

        if not is_pc_request(text):
            return "Я занимаюсь только подбором комплектующих для ПК."

        budget, perf = normalize(text)

        enriched = f"""
Запрос: {text}

budget={budget}
performance={perf}
"""

        self.history.append(
            Messages(role=MessagesRole.USER, content=enriched)
        )

        chat = Chat(
            messages=self.history,
            functions=functions_description,
        )

        response = self.client.chat(chat)
        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "function_call" and message.function_call:

            result = self._run_function(message.function_call)

            self.history.append(
                Messages(
                    role=MessagesRole.ASSISTANT,
                    content="",
                    function_call=message.function_call,
                )
            )

            # function result
            self.history.append(
                Messages(
                    role=MessagesRole.FUNCTION,
                    name=message.function_call.name,
                    content=json.dumps(result, ensure_ascii=False),
                )
            )

            # final answer
            final = self.client.chat(Chat(messages=self.history))
            answer = final.choices[0].message.content

        else:
            answer = message.content

        self.history.append(
            Messages(role=MessagesRole.ASSISTANT, content=answer)
        )

        return answer


def main():
    print("=" * 60)
    print(" Консультат Глеб Павлович Терентьев ")
    print("=" * 60)

    credentials = os.getenv("GIGACHAT_CREDENTIALS")

    if not credentials:
        print("Нет GIGACHAT_CREDENTIALS")
        return

    agent = PCAgent(credentials)

    while True:
        text = input("\nТы: ").strip()

        if text.lower() in ["exit", "quit", "выход"]:
            break

        print("\n... думаю ...")

        try:
            print("\nАссистент:\n", agent.ask(text))
        except Exception as e:
            print("Ошибка:", e)


if __name__ == "__main__":
    main()
