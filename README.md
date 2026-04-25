Track A
---
AI Agent - PC Building Consultant "Gleb Pavlovich Terentyev"

Works with a JSON file containing graphics cards and CPUs, each with a cost and a conditional performance coefficient. The file also includes a list of games with their required performance coefficients.

## Features

- Selects the minimum required PC for a specified game
- Selects the most powerful PC for a given budget
- Selects the PC that best matches a specified performance target
- Can maintain a conversation on electronics topics

---

## Usage

# 1. Install dependencies
pip install gigachat python-dotenv

# 2. Add your auth key in .env
GIGACHAT_CREDENTIALS=enteryourkey

# 3. Launch!
python agent.py
