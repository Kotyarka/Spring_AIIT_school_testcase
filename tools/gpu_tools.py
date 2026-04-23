import json
from langchain.tools import tool
from typing import Dict, Any

def load_data() -> Dict:
    with open("data/components.json", "r") as f:
        return json.load(f)

@tool
def find_gpu_by_performance_and_price(target_performance: float, max_price: float) -> Dict[str, Any]:
    """
    Находит видеокарту, которая соответствует производительности и бюджету
    
    Args:
        target_performance: условный коэффициент производительности
        max_price: Максимальная цена (в у.е.)
    
    Returns:
        Dict с информацией о подобранной видеокарте
    """
    data = load_data()
    gpus = data["gpus"]
    
    affordable_gpus = [gpu for gpu in gpus if gpu["price"] <= max_price]
    
    if not affordable_gpus:
        return {"error": "Видеокарт в данном ценовом диапазоне нет, спасибо нейросетям"}
    
    best_gpu = min(affordable_gpus,
                   key=lambda gpu: abs(gpu["performance"] - target_performance))
    
    max_perf_gpu = max(affordable_gpus, key=lambda gpu: gpu["performance"])
    
    return {
        "selected": best_gpu,
        "max_performance_option": max_perf_gpu,
        "target_performance": target_performance,
        "max_price": max_price
    }
