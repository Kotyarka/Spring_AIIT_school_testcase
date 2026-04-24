import json
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters

load_dotenv()

def load_data() -> Dict:
    with open("data/components.json", "r", encoding="utf-8") as f:
        return json.load(f)

def find_cpu_by_performance_and_price(target_performance: float, max_price: float) -> Dict[str, Any]:

    data = load_data()
    cpus = data["cpus"]
    
    affordable_cpus = [cpu for cpu in cpus if cpu["price"] <= max_price]
    
    if not affordable_cpus:
        return {
            "error": f"Нет процессоров в бюджете ${max_price}", 
            "available": False,
            "message": "В базе данных нет процессоров в указанном бюджете"
        }
    
    best_cpu = min(affordable_cpus, 
                   key=lambda cpu: abs(cpu["performance"] - target_performance))
    
    return {
        "selected": best_cpu,
        "target_performance": target_performance,
        "max_price": max_price
    }

def find_gpu_by_performance_and_price(target_performance: float, max_price: float) -> Dict[str, Any]:
    data = load_data()
    gpus = data["gpus"]
    
    affordable_gpus = [gpu for gpu in gpus if gpu["price"] <= max_price]
    
    if not affordable_gpus:
        return {"error": "Нет видеокарт в заданном бюджете",
                "available": False,
                "message": "В базе данных нет видеокарт в указанном бюджете"
        }
    
    best_gpu = min(affordable_gpus,
                   key=lambda gpu: abs(gpu["performance"] - target_performance))
    
    return {
        "selected": best_gpu,
        "target_performance": target_performance,
        "max_price": max_price
    }

AVAILABLE_FUNCTIONS = {
    "find_cpu_by_performance_and_price": find_cpu_by_performance_and_price,
    "find_gpu_by_performance_and_price": find_gpu_by_performance_and_price,
}

functions_description = [
    Function(
        name="find_cpu_by_performance_and_price",
        description="Находит процессор по целевой производительности и максимальной цене",
        parameters=FunctionParameters(
            type="object",
            properties={
                "target_performance": {
                    "type": "number",
                    "description": "Целевой коэффициент производительности процессора (например, 1.5)"
                },
                "max_price": {
                    "type": "number",
                    "description": "Максимальная цена в долларах США"
                }
            },
            required=["target_performance", "max_price"],
        ),
    ),
    Function(
        name="find_gpu_by_performance_and_price",
        description="Находит видеокарту по целевой производительности и максимальной цене",
        parameters=FunctionParameters(
            type="object",
            properties={
                "target_performance": {
                    "type": "number",
                    "description": "Целевой коэффициент производительности видеокарты (например, 1.5)"
                },
                "max_price": {
                    "type": "number",
                    "description": "Максимальная цена в долларах США"
                }
            },
            required=["target_performance", "max_price"],
        ),
    ),
]


SYSTEM_PROMPT = """У тебя есть доступ к функциям:
1. find_cpu_by_performance_and_price - для подбора процессора
2. find_gpu_by_performance_and_price - для подбора видеокарты

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. ПЕРЕД использованием любой функции ты ДОЛЖЕН проверить реалистичность запроса:
   - Для сборки ПК под игры минимальный бюджет составляет $500
   - При бюджете 10000 рублей и игре "Ведьмак" (The Witcher) - это нереалистичный запрос. Игра стоит около 2000 рублей, а ПК за 10000 рублей - это избыточно дешево.
   
2. ЕСЛИ запрос пользователя нереалистичен или подходящих компонентов нет в базе данных:
   - НЕ вызывай функции find_cpu_by_performance_and_price или find_gpu_by_performance_and_price
   - Сразу скажи пользователю, что не можешь подобрать компоненты
   - Объясни причину: бюджет слишком мал/велик или компонентов нет в базе
   
3. ВЫЗЫВАЙ функции ТОЛЬКО когда:
   - Бюджет находится в разумных пределах ($500 - $3000 для игрового ПК)
   - Есть реальная возможность найти компоненты в базе данных
   
4. Если результат функции вернул {"available": false} или {"error": ...}:
   - НЕ выдумывай компоненты
   - Сообщи пользователю, что подходящих компонентов нет в базе
   - Предложи увеличить бюджет или ознакомиться с доступным ассортиментом

5. Для игрового ПК распределение производительности обычно: 40% на CPU, 60% на GPU.
6. Используй ТОЛЬКО компоненты из результатов функций, не выдумывай свои.

Будь вежлив и полезен. Если не можешь помочь с запросом, прямо скажи об этом."""

class PCAssistantAgent:
    def __init__(self, credentials: str):
        self.client = GigaChat(
            credentials=credentials,
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            temperature=0.7,
            max_tokens=2000,
        )
        self.history = [
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT)
        ]
    
    def _execute_function_call(self, function_call) -> str:
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        
        print(f"\n Вызов функции: {function_name}")
        print(f" Аргументы: {function_args}")
        
        if function_name in AVAILABLE_FUNCTIONS:
            result = AVAILABLE_FUNCTIONS[function_name](**function_args)
            print(f" Результат: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return json.dumps(result, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Неизвестная функция: {function_name}"})
    
    def _send_request(self, user_message: str, has_functions: bool = True) -> Any:
        self.history.append(Messages(role=MessagesRole.USER, content=user_message))
        
        chat = Chat(
            messages=self.history,
            functions=functions_description if has_functions else None,
        )
        
        return self.client.chat(chat)
    
    def _process_function_calls(self, response) -> str:
        """
        Обрабатывает цепочку function calls с проверкой наличия данных.
        """
        choice = response.choices[0]
        message = choice.message
        
        if choice.finish_reason != "function_call":
            return message.content
        
        if message.function_call:
            function_result = self._execute_function_call(message.function_call)
            result_dict = json.loads(function_result)
            
            # Проверяем, есть ли данные
            if result_dict.get("no_data"):
                # Возвращаем сообщение, что данных нет
                error_msg = result_dict.get("message", "Нет подходящих компонентов в базе данных")
                self.history.append(Messages(
                    role=MessagesRole.ASSISTANT,
                    content=f"❌ {error_msg}"
                ))
                return f"❌ {error_msg}"
            
            self.history.append(Messages(
                role=MessagesRole.ASSISTANT,
                content=message.content,
                function_call=message.function_call
            ))
            self.history.append(Messages(
                role=MessagesRole.FUNCTION,
                content=function_result,
                name=message.function_call.name
            ))
            
            chat = Chat(messages=self.history)
            new_response = self.client.chat(chat)
            return new_response.choices[0].message.content
        
        return "Не удалось обработать ответ модели"
    def ask(self, user_message: str) -> str:
        print(f"\nПользователь: {user_message}")
        
        response = self._send_request(user_message)
        answer = self._process_function_calls(response)
        
        self.history.append(Messages(role=MessagesRole.ASSISTANT, content=answer))
        
        return answer
    
    def clear_history(self):
        self.history = [Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT)]


def main():
    print("=" * 60)
    print(" Консультант Глеб Павлович Терентьев ")
    print("=" * 60)
    print("Я помогу подобрать процессор и видеокарту под твой бюджет и требования.")
    print("Напиши, например: 'подбери комплектующие с производительностью 3 и бюджетом 800 долларов'")
    print("Для выхода напиши 'выход' или 'quit'")
    print("-" * 60)
    
    credentials=os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        print(" Ошибка: не найден GIGACHAT_CREDENTIALS в .env файле")
        return
    
    agent = PCAssistantAgent(credentials)
    
    while True:
        user_input = input("\nТы: ").strip()
        
        if user_input.lower() in ['выход', 'quit', 'exit']:
            print("До свидания! Удачной сборки!")
            break
        
        if not user_input:
            continue
        
        print("\n Думаю...")
        try:
            response = agent.ask(user_input)
            print(f"\nАссистент: {response}")
        except Exception as e:
            print(f"\nОшибка: {e}")

if __name__ == "__main__":
    main()
