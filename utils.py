import math


def is_prime(n: int) -> bool:
    """Проверка простого числа."""
    return n >= 2 and all(n % i != 0 for i in range(2, int(math.isqrt(n)) + 1))


def factorial(n: int) -> int:
    """Вычисление факториала."""
    if n < 0:
        raise ValueError("Факториал отрицательного числа не определён")
    return math.prod(range(1, n + 1), start=1)


def gcd(a: int, b: int) -> int:
    """Нахождение НОД (алгоритм Евклида)."""
    return abs(a) if b == 0 else gcd(b, a % b)


def celsius_to_kelvin(c: float) -> float:
    """Конвертация Цельсий → Кельвин."""
    return c + 273.15


def is_palindrome(s: str) -> bool:
    """Проверка строки на палиндром."""
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]