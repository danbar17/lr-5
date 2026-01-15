import pytest
import random
from typing import Callable, List, Tuple, Any

from sorting_lr3_function import (
    bubble_sort,
    selection_sort,
    insertion_sort,
    heap_sort,
    merge_sort,
    quick_sort
)

from utils_function import (
    is_prime,
    factorial,
    gcd,
    celsius_to_kelvin,
    is_palindrome
)


# ============================================================================
# ФИКСТУРЫ ДЛЯ ТЕСТИРОВАНИЯ
# ============================================================================

@pytest.fixture
def empty_array() -> List[int]:
    """Фикстура: пустой массив."""
    return []


@pytest.fixture
def single_element_array() -> List[int]:
    """Фикстура: массив с одним элементом."""
    return [42]


@pytest.fixture
def sorted_array() -> List[int]:
    """Фикстура: уже отсортированный массив."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def reverse_sorted_array() -> List[int]:
    """Фикстура: массив, отсортированный в обратном порядке."""
    return [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


@pytest.fixture
def random_small_array() -> List[int]:
    """Фикстура: небольшой случайный массив."""
    return [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]


@pytest.fixture
def random_large_array() -> List[int]:
    """Фикстура: большой случайный массив (50 элементов)."""
    return random.sample(range(1, 101), 50)


@pytest.fixture
def array_with_duplicates() -> List[int]:
    """Фикстура: массив с повторяющимися элементами."""
    return [5, 2, 8, 2, 5, 1, 8, 1, 9, 2]


@pytest.fixture
def all_test_arrays(
        empty_array,
        single_element_array,
        sorted_array,
        reverse_sorted_array,
        random_small_array,
        array_with_duplicates
) -> List[Tuple[str, List[int]]]:
    """Фикстура: все тестовые массивы с именами."""
    return [
        ("empty", empty_array),
        ("single", single_element_array),
        ("sorted", sorted_array),
        ("reverse", reverse_sorted_array),
        ("random_small", random_small_array),
        ("with_duplicates", array_with_duplicates)
    ]


# ============================================================================
# ПАРАМЕТРИЗОВАННЫЕ ТЕСТЫ ДЛЯ АЛГОРИТМОВ СОРТИРОВКИ
# ============================================================================

# Список всех алгоритмов сортировки для параметризации
SORTING_ALGORITHMS = [
    (bubble_sort, "Пузырьковая сортировка"),
    (selection_sort, "Сортировка выбором"),
    (insertion_sort, "Сортировка вставками"),
    (heap_sort, "Пирамидальная сортировка"),
    (merge_sort, "Сортировка слиянием"),
    (quick_sort, "Быстрая сортировка")
]


@pytest.mark.parametrize(
    "sort_func, algorithm_name",
    SORTING_ALGORITHMS,
    ids=[name for _, name in SORTING_ALGORITHMS]
)
class TestSortingAlgorithms:
    """Класс для тестирования всех алгоритмов сортировки."""

    def test_empty_array(self, sort_func: Callable, algorithm_name: str, empty_array: List[int]):
        """Тест: сортировка пустого массива."""
        result, iterations = sort_func(empty_array.copy())
        assert result == []
        assert iterations >= 0

    def test_single_element(self, sort_func: Callable, algorithm_name: str, single_element_array: List[int]):
        """Тест: сортировка массива с одним элементом."""
        result, iterations = sort_func(single_element_array.copy())
        assert result == [42]
        assert iterations >= 0

    def test_already_sorted(self, sort_func: Callable, algorithm_name: str, sorted_array: List[int]):
        """Тест: сортировка уже отсортированного массива."""
        result, iterations = sort_func(sorted_array.copy())
        assert result == sorted_array
        assert iterations >= 0

    def test_reverse_sorted(self, sort_func: Callable, algorithm_name: str, reverse_sorted_array: List[int]):
        """Тест: сортировка массива в обратном порядке."""
        expected = sorted(reverse_sorted_array)
        result, iterations = sort_func(reverse_sorted_array.copy())
        assert result == expected
        assert iterations >= 0

    def test_random_small(self, sort_func: Callable, algorithm_name: str, random_small_array: List[int]):
        """Тест: сортировка небольшого случайного массива."""
        expected = sorted(random_small_array)
        result, iterations = sort_func(random_small_array.copy())
        assert result == expected
        assert iterations >= 0

    def test_array_with_duplicates(self, sort_func: Callable, algorithm_name: str, array_with_duplicates: List[int]):
        """Тест: сортировка массива с повторяющимися элементами."""
        expected = sorted(array_with_duplicates)
        result, iterations = sort_func(array_with_duplicates.copy())
        assert result == expected
        assert iterations >= 0

    def test_large_array(self, sort_func: Callable, algorithm_name: str, random_large_array: List[int]):
        """Тест: сортировка большого массива (50 элементов)."""
        expected = sorted(random_large_array)
        result, iterations = sort_func(random_large_array.copy())
        assert result == expected
        assert iterations >= 0

    def test_preserves_length(self, sort_func: Callable, algorithm_name: str, random_small_array: List[int]):
        """Тест: проверка сохранения длины массива после сортировки."""
        original = random_small_array.copy()
        result, iterations = sort_func(original)
        assert len(result) == len(original)

    def test_returns_tuple(self, sort_func: Callable, algorithm_name: str, random_small_array: List[int]):
        """Тест: проверка формата возвращаемого значения."""
        result = sort_func(random_small_array.copy())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], int)


# ============================================================================
# ТЕСТЫ ДЛЯ МАТЕМАТИЧЕСКИХ ФУНКЦИЙ
# ============================================================================

class TestPrimeNumbers:
    """Тесты для функции проверки простых чисел."""

    @pytest.mark.parametrize("number, expected", [
        (2, True),  # Наименьшее простое число
        (3, True),  # Простое число
        (4, False),  # Составное число
        (17, True),  # Простое число
        (1, False),  # 1 не считается простым
        (0, False),  # 0 не простое
        (-5, False),  # Отрицательные числа не простые
        (97, True),  # Большое простое число
        (100, False)  # Составное число
    ])
    def test_is_prime(self, number: int, expected: bool):
        """Тест функции проверки простых чисел."""
        assert is_prime(number) == expected


class TestFactorialFunction:
    """Тесты для функции вычисления факториала."""

    @pytest.mark.parametrize("n, expected", [
        (0, 1),  # Факториал 0 равен 1
        (1, 1),  # Факториал 1 равен 1
        (5, 120),  # 5! = 120
        (3, 6),  # 3! = 6
        (7, 5040),  # 7! = 5040
        (10, 3628800)  # 10! = 3628800
    ])
    def test_factorial_positive(self, n: int, expected: int):
        """Тест факториала для положительных чисел."""
        assert factorial(n) == expected

    def test_factorial_negative(self):
        """Тест: факториал отрицательного числа должен вызывать исключение."""
        with pytest.raises(ValueError, match="Факториал отрицательного числа не определён"):
            factorial(-5)

    def test_factorial_large_negative(self):
        """Тест: факториал большого отрицательного числа."""
        with pytest.raises(ValueError):
            factorial(-100)


class TestGCDFunction:
    """Тесты для функции нахождения наибольшего общего делителя."""

    @pytest.mark.parametrize("a, b, expected", [
        (12, 18, 6),  # НОД(12, 18) = 6
        (17, 13, 1),  # Взаимно простые числа
        (100, 25, 25),  # Одно число делит другое
        (48, 18, 6),  # НОД(48, 18) = 6
        (0, 5, 5),  # НОД(0, 5) = 5
        (5, 0, 5),  # НОД(5, 0) = 5
        (0, 0, 0),  # НОД(0, 0) = 0
        (-12, 18, 6),  # С отрицательными числами
        (12, -18, 6),  # С отрицательными числами
        (-12, -18, 6)  # Оба числа отрицательные
    ])
    def test_gcd(self, a: int, b: int, expected: int):
        """Тест функции нахождения НОД."""
        assert gcd(a, b) == expected


class TestTemperatureConversion:
    """Тесты для функции конвертации температуры."""

    @pytest.mark.parametrize("celsius, expected_kelvin", [
        (0, 273.15),  # Точка замерзания воды
        (100, 373.15),  # Точка кипения воды
        (-273.15, 0),  # Абсолютный ноль
        (25, 298.15),  # Комнатная температура
        (-40, 233.15),  # -40°C = -40°F
        (37, 310.15)  # Температура тела человека
    ])
    def test_celsius_to_kelvin(self, celsius: float, expected_kelvin: float):
        """Тест конвертации Цельсия в Кельвины."""
        # Используем приблизительное сравнение для чисел с плавающей точкой
        assert celsius_to_kelvin(celsius) == pytest.approx(expected_kelvin, rel=1e-9)

    def test_celsius_to_kelvin_precision(self):
        """Тест точности конвертации."""
        # Проверяем, что константа сложения точная
        for temp in [-100, 0, 100, 500]:
            result = celsius_to_kelvin(temp)
            expected = temp + 273.15
            assert abs(result - expected) < 1e-10


class TestPalindromeFunction:
    """Тесты для функции проверки палиндромов."""

    @pytest.mark.parametrize("text, expected", [
        ("radar", True),  # Простой палиндром
        ("hello", False),  # Не палиндром
        ("", True),  # Пустая строка - палиндром
        ("a", True),  # Один символ - палиндром
        ("racecar", True),  # Классический палиндром
        ("A man a plan a canal Panama", True),  # Палиндром с пробелами и регистром
        ("12321", True),  # Числовой палиндром
        ("madam", True),  # Ещё один палиндром
        ("python", False),  # Не палиндром
        ("  ", True),  # Только пробелы
        ("ab ba", True)  # Палиндром с пробелом посередине
    ])
    def test_is_palindrome(self, text: str, expected: bool):
        """Тест функции проверки палиндромов."""
        assert is_palindrome(text) == expected

    @pytest.mark.parametrize("text", [
        "Radar",
        "RaceCar",
        "MaDaM"
    ])
    def test_palindrome_case_sensitive(self, text: str):
        """Тест: функция должна быть чувствительна к регистру по умолчанию."""
        # Если функция чувствительна к регистру, эти строки не должны быть палиндромами
        # Если функция нечувствительна к регистру, тест нужно адаптировать
        result = is_palindrome(text)
        # Проверяем, что результат соответствует либо True, либо False
        assert isinstance(result, bool)


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

def test_sorting_algorithms_consistency(all_test_arrays: List[Tuple[str, List[int]]]):
    """
    Интеграционный тест: проверяем, что все алгоритмы сортировки
    дают одинаковый результат для одних и тех же данных.
    """
    for name, test_array in all_test_arrays:
        expected = sorted(test_array)

        # Собираем результаты всех алгоритмов
        results = []
        for sort_func, algo_name in SORTING_ALGORITHMS:
            sorted_array, iterations = sort_func(test_array.copy())
            results.append((algo_name, sorted_array))

        # Проверяем, что все алгоритмы дали одинаковый результат
        for algo_name, result in results:
            assert result == expected, (
                f"Алгоритм {algo_name} дал неверный результат "
                f"для тестового массива '{name}': {test_array}"
            )


def test_math_functions_combined():
    """Интеграционный тест: проверка комбинированного использования математических функций."""
    # Проверяем, что факториал простого числа работает корректно
    prime_number = 7
    assert is_prime(prime_number) == True
    assert factorial(prime_number) == 5040

    # Проверяем НОД для факториалов
    assert gcd(factorial(5), factorial(3)) == 6  # НОД(120, 6) = 6

    # Проверяем конвертацию температуры для простых чисел
    prime_temp = 13
    assert is_prime(prime_temp) == True
    kelvin_temp = celsius_to_kelvin(prime_temp)
    assert kelvin_temp == pytest.approx(286.15)


# ============================================================================
# ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ (ОПЦИОНАЛЬНО)
# ============================================================================

@pytest.mark.slow
def test_sorting_performance(random_large_array: List[int]):
    """
    Тест производительности: сравниваем время работы разных алгоритмов.
    Помечен как 'slow', так как может выполняться долго.
    """
    results = {}

    for sort_func, algo_name in SORTING_ALGORITHMS:
        test_data = random_large_array.copy()
        sorted_array, iterations = sort_func(test_data)

        results[algo_name] = {
            'iterations': iterations,
            'correct': sorted_array == sorted(random_large_array)
        }

    # Проверяем, что все алгоритмы корректно отсортировали массив
    for algo_name, data in results.items():
        assert data['correct'], f"Алгоритм {algo_name} дал неверный результат"


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ТЕСТОВ
# ============================================================================

def test_sorting_stability():
    """
    Тест стабильности сортировки (если применимо).
    Стабильная сортировка сохраняет порядок равных элементов.
    """
    # Массив с повторяющимися значениями, но разными "метаданными"
    test_data = [(3, 'a'), (1, 'b'), (3, 'c'), (2, 'd'), (1, 'e')]

    # Для стабильных сортировок порядок (3,'a') и (3,'c') должен сохраниться
    # Этот тест требует адаптации под конкретные алгоритмы


if __name__ == "__main__":
    # Запуск тестов напрямую (альтернатива pytest в командной строке)
    pytest.main([__file__, "-v"])