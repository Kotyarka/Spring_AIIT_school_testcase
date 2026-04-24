import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters

load_dotenv()


def load_data() -> Dict:
    with open("data/components.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_game_performance(game_name: str, data: Dict) -> float | None:
    if not game_name:
        return None

    for g in data.get("games", []):
        if g["name"].lower() == game_name.lower():
            return g["required_performance"]

    return None
# ===================== функции =====================

def get_game_performance(game_name: str, data: Dict) -> float | None:
    if not game_name:
        return None

    for g in data.get("games", []):
        if g["name"].lower() == game_name.lower():
            return g["required_performance"]

    return None

def build_pc(target_performance: float = None, budget: float = None, is_game_based: bool = False):
    data = load_data()

    cpus = data["cpus"]
    gpus = data["gpus"]

    best_combo = None

    #  указана только стоимость - ищем максимальную производительность
    if budget is not None and target_performance is None:
        best_performance = -1
        
        for cpu in cpus:
            for gpu in gpus:
                total_price = cpu["price"] + gpu["price"]
                if total_price > budget:
                    continue
                
                performance = cpu["performance"] + gpu["performance"]
                
                if performance > best_performance:
                    best_performance = performance
                    best_combo = (cpu, gpu, total_price, performance)
        
        if not best_combo:
            cpu = min(cpus, key=lambda x: x["price"])
            gpu = min(gpus, key=lambda x: x["price"])
            return {
                "cpu": cpu,
                "gpu": gpu,
                "total_price": cpu["price"] + gpu["price"],
                "performance": cpu["performance"] + gpu["performance"],
                "budget": budget,
                "warning": "Бюджет слишком низкий для сборки"
            }
    
        #   указана только производительность - ищем минимальную стоимость
    elif target_performance is not None and budget is None:
        best_price = float('inf')
        
        for cpu in cpus:
            for gpu in gpus:
                total_price = cpu["price"] + gpu["price"]
                performance = cpu["performance"] + gpu["performance"]
                
                if performance >= target_performance and total_price < best_price:
                    best_price = total_price
                    best_combo = (cpu, gpu, total_price, performance)
        
        if not best_combo:
            cpu = max(cpus, key=lambda x: x["performance"])
            gpu = max(gpus, key=lambda x: x["performance"])
            return {
                "cpu": cpu,
                "gpu": gpu,
                "total_price": cpu["price"] + gpu["price"],
                "performance": cpu["performance"] + gpu["performance"],
                "target_performance": target_performance,
                "warning": "Невозможно достичь заданной производительности"
            }
    
      #   указаны оба параметра - ищем оптимальный баланс
    elif target_performance is not None and budget is not None:
        best_score = -1
        
        for cpu in cpus:
            for gpu in gpus:
                total_price = cpu["price"] + gpu["price"]
                if total_price > budget:
                    continue
                
                performance = cpu["performance"] + gpu["performance"]
                
                perf_score = 1 / (abs(performance - target_performance) + 0.01)
                budget_score = 1 - (total_price / budget)
                score = (perf_score * 0.7) + (budget_score * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_combo = (cpu, gpu, total_price, performance)
        
        if not best_combo:
            cpu = min(cpus, key=lambda x: x["price"])
            gpu = min(gpus, key=lambda x: x["price"])
            return {
                "cpu": cpu,
                "gpu": gpu,
                "total_price": cpu["price"] + gpu["price"],
                "performance": cpu["performance"] + gpu["performance"],
                "budget": budget,
                "warning": "Бюджет слишком низкий для оптимальной сборки"
            }
    
    else:
        raise ValueError("Укажите хотя бы один параметр: budget или target_performance")

    cpu, gpu, total_price, performance = best_combo

    result = {
        "cpu": cpu,
        "gpu": gpu,
        "total_price": round(total_price, 2),
        "performance": round(performance, 2),
        "game_mode": is_game_based
    }
    
    if budget is not None:
        result["budget"] = budget
        result["budget_used_percent"] = round((total_price / budget) * 100, 1)
    
    if target_performance is not None:
        result["target_performance"] = target_performance
    
    return result

AVAILABLE_FUNCTIONS = {
    "build_pc": build_pc,
}


functions_description = [
    Function(
        name="build_pc",
        description="Собирает ПК по бюджету и требуемой производительности",
        parameters=FunctionParameters(
            type="object",
            properties={
                "target_performance": {"type": "number"},
                "budget": {"type": "number"},
                "is_game_based": {"type": "boolean"},
            },
            required=["target_performance", "budget"],
        ),
    )
]


# ===================== Промпты =====================
### Промпт сборки компа
SYSTEM_PROMPT = """
Ты консультант по сборке ПК.

Тебе дают результат сборки компьютера в JSON.

ВСЕГДА ВЫЗЫВАЙ ФУНКЦИЮ СБОРКИ!!! НЕ БЕРИ ДАННЫЕ ИЗ ГОЛОВЫ!!!

Твоя задача:
- красиво объяснить сборку
- вывести:

CPU: {name}
GPU: {name}
Общая цена: {price}
Производительность: {performance} в коэффициенте производительности

Говори кратко и понятно.
"""
### Промпт парсера данных
PARSER_PROMPT = """
Ты — парсер запросов о ПК.

Определи параметры из текста пользователя.

Верни ТОЛЬКО JSON:

{
  "budget": number | null,
  "target_performance": number | null,
  "game": string | null,
  "is_pc_request": boolean
}

Если данных нет → null
"""
### Промпт балабола о технике
TECH_PROMPT = """
Ты технический ассистент.

Отвечай ТОЛЬКО на темы:
- компьютеры
- железо (CPU, GPU, RAM, SSD)
- ноутбуки
- смартфоны
- периферия
- софт и технологии

Если вопрос не про технику — скажи:
"Я отвечаю только на вопросы о технике."

Отвечай кратко и по делу.
"""
# ===================== Агент =====================

class PCAgent:
    def __init__(self, credentials: str):
        self.client = GigaChat(
            credentials=credentials,
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            temperature=0
        )

        self.history = [
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT)
        ]

    def parse_user_intent(self, text: str) -> Dict[str, Any]:
        response = self.client.chat(Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=PARSER_PROMPT),
                Messages(role=MessagesRole.USER, content=text)
            ],
            temperature=0
        ))

        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {
                "budget": None,
                "target_performance": None,
                "game": None,
                "is_pc_request": False
            }

    def _run_function(self, function_call, params):
        name = function_call.name
        result = AVAILABLE_FUNCTIONS[name](**params)
        return result

    def normalize(self, parsed: dict) -> dict:
        data = load_data()

        game = parsed.get("game")
        target = parsed.get("target_performance")

        if target is None and game:
            target = None
            for g in data.get("games", []):
                if g["name"].lower() == game.lower():
                    target = g["required_performance"]
                    break

        return {
            "budget": parsed.get("budget") or 800,
            "performance": target or 2.0, 
            "game": game,
            "is_game_based": game is not None
        }

    # -------- main --------
    def ask(self, text: str):

        parsed = self.parse_user_intent(text)

        # ---- Сборка пк ----
        if parsed.get("is_pc_request"):
            params = self.normalize(parsed)

            chat = Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
                    Messages(role=MessagesRole.USER, content=json.dumps(params))
                ],
                functions=functions_description
            )

            response = self.client.chat(chat)
            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason == "function_call":
                result = self._run_function(
                    message.function_call,
                    {
                        "target_performance": params["performance"],
                        "budget": params["budget"],
                        "is_game_based": params["is_game_based"]
                    }
                )

                final = self.client.chat(Chat(
                    messages=[
                        Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
                        Messages(
                            role=MessagesRole.USER,
                            content=json.dumps(result, ensure_ascii=False)
                        )
                    ]
                ))

                return final.choices[0].message.content

            return message.content

        # ---- Балабольство ----
        tech_response = self.client.chat(Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=TECH_PROMPT),
                Messages(role=MessagesRole.USER, content=text)
            ],
            temperature=0.3
        ))

        return tech_response.choices[0].message.content

# ===================== MAIN =====================

def main():
    print("=" * 60)
    print(" Консультант Глеб Павлович Терентьев ")
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

        print("\nГ.П.Терентьев думает...\n")
        try:
            print(agent.ask(text))
        except Exception as e:
            print("Ошибка:", e)


if __name__ == "__main__":
    main()
