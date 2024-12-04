import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-10, 10, 0.1)
w = 5
b = 3
y = w * x + b

plt.figure(figsize=(16,8))
plt.plot(x, y, label = ' y = wx + b')
plt.title('Function: y = wx + b')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
#2nd
y = x ** 2

plt.figure(figsize=(16,8))
plt.plot(x, y, label = ' y = x^2')
plt.title('Function: y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
#3rd
y = 1 / (1 + np.exp(-x))

plt.figure(figsize=(16,8))
plt.plot(x, y, label = ' y = 1 / (1 + e^-x)')
plt.title('Function: y = 1 / (1 + e^-x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
#4rth
y = (np.exp(x) - np.exp(-x)) / (np.exp(x) - np.exp(-x))

plt.figure(figsize=(16,8))
plt.plot(x, y, label = ' y = (e^x - e^-x) / (e^x + e^-x)')
plt.title('Function: y = (e^x - e^-x) / (e^x + e^-x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
#5th
u = w * x + b
y = 1 / (1 + np.exp(-u))

plt.figure(figsize=(16,8))
plt.plot(x, y, label = ' y = g(f(x))')
plt.title('Function: y = g(f(x))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
#6th
u = w * x + b
y = (np.exp(u) - np.exp(-u)) / (np.exp(u) + np.exp(-u))

plt.figure(figsize=(16,8))
plt.plot(x, y, label = 'y = g(f(x))')
plt.title('Function: y = g(f(x))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
#7th
w1, b1 = 4, 10
u1 = w1 * x + b1
y1 = 1 / (1 + np.exp(-u1))

w2, b2 = 6, 3
u2 = w2 * x + b2
y2 = 1 / (1 + np.exp(-u2))

w3, w4 = 2, 3
w = w3 * y1 + w4 * y2 + b
y = 1 / (1 + np.exp(-w))

plt.figure(figsize=(16, 8))
plt.plot(x, y, label = 'y = g3(f3(v))')
plt.title('Function: y = g3(f3(v))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
