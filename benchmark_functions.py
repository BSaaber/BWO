import numpy as np


# Main benchmark functions from the original paper

def powell_sum(x):
    """
    F1: Powell Sum (Some of different powers)
    f(x) = sum_{i=1}^n |x_i|^(i+1)
    
    Global minimum: f(0,0,...,0) = 0
    Search domain: [-5.12, 5.12]^n
    """
    n = len(x)
    return np.sum([np.abs(x[i])**(i+2) for i in range(n)])


def cigar(x):
    """
    F2: Cigar function
    f(x) = x_1^2 + 10^6 * sum_{i=2}^n x_i^2
    
    Global minimum: f(0,0,...,0) = 0
    Search domain: [-5.12, 5.12]^n
    """
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)


def discus(x):
    """
    F3: Discus function
    f(x) = 10^6 * x_1^2 + sum_{i=2}^n x_i^2
    
    Global minimum: f(0,0,...,0) = 0
    Search domain: [-5.12, 5.12]^n
    """
    return 1e6 * x[0]**2 + np.sum(x[1:]**2)


def rosenbrock(x):
    """
    F4: Rosenbrock function
    f(x) = sum_{i=1}^{n-1} [100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2]
    
    Global minimum: f(1,1,...,1) = 0
    Search domain: [-30, 30]^n
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


def ackley(x):
    """
    F5: Ackley function
    f(x) = -20*exp(-0.2*sqrt(1/n * sum_{i=1}^n x_i^2)) 
           - exp(1/n * sum_{i=1}^n cos(2*pi*x_i)) + 20 + e
    
    Global minimum: f(0,0,...,0) = 0
    Search domain: [-35, 35]^n
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + 20 + np.exp(1)


# Additional benchmark functions

def sphere(x):
    """
    Sphere function
    f(x) = sum_{i=1}^n x_i^2
    
    Global minimum: f(0,0,...,0) = 0
    Typical search domain: [-5.12, 5.12]^n
    """
    return np.sum(x**2)


def rastrigin(x):
    """
    Rastrigin function
    f(x) = 10n + sum_{i=1}^n [x_i^2 - 10*cos(2*pi*x_i)]
    
    Global minimum: f(0,0,...,0) = 0
    Typical search domain: [-5.12, 5.12]^n
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def griewank(x):
    """
    Griewank function
    f(x) = 1 + sum_{i=1}^n x_i^2/4000 - prod_{i=1}^n cos(x_i/sqrt(i))
    
    Global minimum: f(0,0,...,0) = 0
    Typical search domain: [-600, 600]^n
    """
    n = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
    
    return 1 + sum_term - prod_term


def schwefel(x):
    """
    Schwefel function
    f(x) = 418.9829*n - sum_{i=1}^n [x_i * sin(sqrt(|x_i|))]
    
    Global minimum: f(420.9687,...,420.9687) = 0
    Typical search domain: [-500, 500]^n
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy(x):
    """
    Levy function
    
    Global minimum: f(1,...,1) = 0
    Typical search domain: [-10, 10]^n
    """
    n = len(x)
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    
    return term1 + term2 + term3
