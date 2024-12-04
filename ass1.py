import numpy as np
import matplotlib.pyplot as plt
import random
def linear_func(a, b, x):
    return a * x + b

def sin_func(a, b, x):
    return a * np.sin(x) + b

def quadratic_func(a, b, x):
    return a * x**2 + b

def exponential_func(a, b, x):
    return a * np.exp(x) + b
a = random.randint(1, 5)
b = random.randint(1, 5)
x = np.linspace(-10, 10, 100)

print("a = ", a , "b = ", b)
y_linear = linear_func(a, b, x)
y_sin = sin_func(a, b, x)
y_quadratic = quadratic_func(a, b, x)
y_exponential = exponential_func(a, b, x)

plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(x, y_linear, marker='o')
plt.title('Linear function')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(x, y_sin, marker='o')
plt.title('Sinusoidal function (non-linear)')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(x, y_quadratic, marker='o')
plt.title('Quadratic function (non-linear)')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(x, y_exponential, marker='o')
plt.title('Exponential function (non-linear)')
plt.grid()

plt.tight_layout(pad=3.0)
plt.show()

x = np.linspace(-10, 10, 10)

y_linear = linear_func(a, b, x)
y_sin = sin_func(a, b, x)
y_quadratic = quadratic_func(a, b, x)
y_exponential = exponential_func(a, b, x)

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_linear, marker='o')
for i, j in zip(x, y_linear):
    plt.annotate(f'({i:.2f}, {j:.2f})', (i, j))
plt.title('Linear function')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, y_sin, marker='o')
for i, j in zip(x, y_sin):
    plt.annotate(f'({i:.2f}, {j:.2f})', (i, j))
plt.title('Sinusoidal function (non-linear)')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(x, y_quadratic, marker='o')
for i, j in zip(x, y_quadratic):
    plt.annotate(f'({i:.2f}, {j:.2f})', (i, j))
plt.title('Quadratic function (non-linear)')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(x, y_exponential, marker='o')
for i, j in zip(x, y_exponential):
    plt.annotate(f'({i:.2f}, {j:.2f})', (i, j))
plt.title('Exponential function (non-linear)')
plt.grid()

plt.tight_layout(pad=3.0)
plt.show()