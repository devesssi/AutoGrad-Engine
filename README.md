# Micrograd-like Automatic Differentiation from Scratch

## Project Description

This project implements a basic automatic differentiation engine, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), from scratch using Python. The core component is the `Value` class, which allows for building computational graphs and automatically computing gradients for scalar-valued expressions. This implementation demonstrates the fundamental principles of backpropagation.

## Features

The `Value` class supports the following operations, enabling the construction of simple neural networks and other mathematical expressions:

- **Addition (`+`)**: `__add__`, `__radd__`
- **Multiplication (`*`)**: `__mul__`, `__rmul__`
- **Exponentiation (`**`)**: `__pow__`
- **Division (`/`)**: `__truediv__` (implemented using multiplication and power)
- **Negation (`-`)**: `__neg__` (implemented using multiplication)
- **Subtraction (`-`)**: `__sub__` (implemented using addition and negation)
- **Hyperbolic Tangent (`tanh`)**: `tanh()`
- **Exponential (`exp`)**: `exp()`
- **Automatic Gradient Calculation**: The `backward()` method performs backpropagation to compute gradients for all `Value` objects in the computational graph.
- **Graph Visualization**: Integration with `graphviz` to visualize the computational graph, showing values, operations, and computed gradients.

## How to Use

1.  **Define a `Value` object**:
    ```python
    a = Value(3.0, label='a')
    b = Value(2.0, label='b')
    ```

2.  **Perform operations**:
    ```python
    c = a**2
    d = c / b
    L = d + Value(5.0, label='f')
    ```

3.  **Compute gradients**: Call the `backward()` method on the final `Value` object.
    ```python
    L.backward()
    ```

4.  **Visualize the graph**: Use the `draw_dot()` function to render the computational graph with gradients.
    ```python
    draw_dot(L)
    ```

### Example

```python
import math
from graphviz import Digraph # Assuming graphviz is installed and draw_dot is defined

# Assuming the Value class is defined as in the notebook
# ... (Value class definition)

# Example computational graph
a = Value(3.0, label='a')
b = Value(2.0, label='b')
c = a**2           # Using __pow__
c.label = 'c'
d = c / b          # Using __truediv__
d.label = 'd'
e = -d             # Using __neg__
e.label = 'e'
f = Value(5.0, label='f')
g = e + f          # Using __add__
g.label = 'g'
h = g.tanh()       # Using tanh
h.label = 'h'
i = Value(0.5, label='i')
L = h * i          # Final output, using __mul__
L.label = 'L'

print(f"Final value of L: {L.data}")

# Perform backward pass to compute gradients
L.backward()

# Visualize the graph with computed gradients
draw_dot(L)
This will output the final value of L and render a graph showing the flow of computation and the gradient values for each node.

Dependencies
math
graphviz (for visualization)
numpy (for general numerical operations, not strictly part of Value class but used in examples)
matplotlib (for general plotting, not strictly part of Value class but used in examples)
