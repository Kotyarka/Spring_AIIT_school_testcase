import json
from langchain.tools import tool
from typing import Dict, Any

def load_data() -> Dict:
    with open("data/components.json", "r") as f:
        return json.load(f)

@tool
def find_cpu_by_performance_and_price(target_performance: float, max_price: float) -> Dict[str, Any]:
    """
    Находит процессор, соответствующий цене и проивоздительности
    
    Args:
        target_performance: условный коэффициент производительности
        max_price: максимальная цена в долларах (в у.е.)
    
    Returns:
        Dict с информацией о подобранном процессоре
    """
    data = load_data()
    cpus = data["cpus"]
    
    affordable_cpus = [cpu for cpu in cpus if cpu["price"] <= max_price]
    
    if not affordable_cpus:
        return {"error": "В заданном бюджете процессоров нет, кризис :("}
    
    best_cpu = min(affordable_cpus, 
                   key=lambda cpu: abs(cpu["performance"] - target_performance))
    
    budget_cpu = min(affordable_cpus, key=lambda cpu: cpu["price"])
    
    return {
        "selected": best_cpu,
        "budget_option": budget_cpu,
        "target_performance": target_performance,
        "max_price": max_price
    }
