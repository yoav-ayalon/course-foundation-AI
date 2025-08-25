# course-foundation-AI

**course-foundation-AI**  
Solutions for the Foundations of AI course, focusing only on the first coding task from each assignment.

---

## Contents

- **README.md** – Project overview and instructions.
- **Ex1/q1.py** – Path-finding algorithms for campaigners moving across US counties.
- **Ex3/q3.py** – Decision tree for flight delay prediction.

---

## Implemented Tasks

### Q1 – Path-Finding

**Task:**  
Campaigners move across adjacent US counties, aiming for Red/Blue destinations.

**CSV Input:**  
`adjacency.csv`

**Algorithms Implemented:**  
- A* (`A_star`)
- Hill Climbing (`hill_climbing`)
- Simulated Annealing (`simulated_annealing`)
- K-Beam (`k_beam`)
- Genetic Algorithm (`genetic_algorithm`)

**Usage:**  
- For testing: run `q1.py` directly and follow prompts.
- For learning/copying: review function implementations and usage patterns.

**Example:**
```python
from q1 import read_from_csv, A_star

graph = read_from_csv('adjacency.csv')
start = 'Denver,CO'
goal = 'Miami,FL'
path = A_star(graph, start, goal)
print(path)
```

---

### Q3 – Flight Delay Prediction

**Task:**  
Predict if a flight will be delayed (>15 min) using a decision tree.

**CSV Input:**  
`flightdelay.csv`

**Functions Implemented:**  
- `build_tree`
- `tree_error`
- `is_late`

**Usage:**  
- For testing: run `q3.py` directly.
- For learning/copying: use functions for custom prediction tasks.

**Example:**
```python
from q3 import tree_error, is_late

mean_error = tree_error(5)
row = ['JFK', 'LAX', '2023-06-01', '08:00', 'Delta']
prediction = is_late(row)
print('Late:', prediction)
```

---

## Getting Started

1. Clone the repo:
    ```sh
    git clone https://github.com/yourusername/course-foundation-AI.git
    ```
2. Install dependencies (if any).
3. Place required CSV files in the appropriate folders (`Ex1/`, `Ex3/`).
4. Run example code in `q1.py` or `q3.py`.

---

## License

For educational purposes only.