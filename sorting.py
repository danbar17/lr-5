import math


def bubble_sort(arr):
    """
    Сортировка пузырьком.

    Временная сложность: O(n²) в худшем и среднем случае, O(n) в лучшем случае.

    Args:
        arr (list): Исходный список целых чисел.

    Returns:
        tuple: (отсортированный список, количество обменов элементов)
    """
    if len(arr) <= 1:
        return arr, 0

    n = len(arr)
    swap_count = 0

    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swap_count += 1
                swapped = True

        # Оптимизация: если не было обменов, список отсортирован
        if not swapped:
            break

    return arr, swap_count


def selection_sort(arr):
    """
    Сортировка выбором.

    Временная сложность: O(n²) во всех случаях.

    Args:
        arr (list): Исходный список целых чисел.

    Returns:
        tuple: (отсортированный список, количество итераций внешнего цикла)
    """
    if len(arr) <= 1:
        return arr, 0

    n = len(arr)
    iteration_count = 0

    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j

        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]

        iteration_count += 1

    return arr, iteration_count


def insertion_sort(arr):
    """
    Сортировка вставками.

    Временная сложность: O(n²) в худшем случае, O(n) в лучшем случае.

    Args:
        arr (list): Исходный список целых чисел.

    Returns:
        tuple: (отсортированный список, количество вставок)
    """
    if len(arr) <= 1:
        return arr, 0

    insertion_count = 0

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key
        insertion_count += 1

    return arr, insertion_count


def _heapify(arr, heap_size, root_index):
    """Вспомогательная функция для построения двоичной кучи."""
    largest = root_index
    left_child = 2 * root_index + 1
    right_child = 2 * root_index + 2

    if left_child < heap_size and arr[left_child] > arr[largest]:
        largest = left_child

    if right_child < heap_size and arr[right_child] > arr[largest]:
        largest = right_child

    if largest != root_index:
        arr[root_index], arr[largest] = arr[largest], arr[root_index]
        _heapify(arr, heap_size, largest)


def heap_sort(arr):
    """
    Пирамидальная сортировка (сортировка кучей).

    Временная сложность: O(n log n) во всех случаях.

    Args:
        arr (list): Исходный список целых чисел.

    Returns:
        tuple: (отсортированный список, количество вызовов heapify)
    """
    if len(arr) <= 1:
        return arr, 0

    n = len(arr)
    heapify_count = 0

    # Построение максимальной кучи
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
        heapify_count += 1

    # Извлечение элементов из кучи один за другим
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0)
        heapify_count += 1

    return arr, heapify_count


def _merge_sorted_lists(left, right):
    """
    Слияние двух отсортированных списков в один.

    Args:
        left (list): Первый отсортированный список.
        right (list): Второй отсортированный список.

    Returns:
        list: Объединенный отсортированный список.
    """
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Добавляем оставшиеся элементы
    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


def merge_sort(arr):
    """
    Сортировка слиянием.

    Временная сложность: O(n log n) во всех случаях.

    Args:
        arr (list): Исходный список целых чисел.

    Returns:
        tuple: (отсортированный список, количество рекурсивных вызовов)
    """
    if len(arr) <= 1:
        return arr, 0

    # Разделение списка пополам
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Рекурсивная сортировка половинок
    left_sorted, left_calls = merge_sort(left_half)
    right_sorted, right_calls = merge_sort(right_half)

    # Слияние отсортированных половинок
    result = _merge_sorted_lists(left_sorted, right_sorted)
    total_calls = 1 + left_calls + right_calls

    return result, total_calls


def quick_sort(arr):
    """
    Быстрая сортировка.

    Временная сложность: O(n log n) в среднем случае, O(n²) в худшем случае.

    Args:
        arr (list): Исходный список целых чисел.

    Returns:
        tuple: (отсортированный список, количество рекурсивных вызовов)
    """
    if len(arr) <= 1:
        return arr, 0

    # Выбор опорного элемента (первый элемент)
    pivot = arr[0]

    # Разделение на подсписки
    less_than_pivot = [x for x in arr[1:] if x < pivot]
    equal_to_pivot = [x for x in arr if x == pivot]
    greater_than_pivot = [x for x in arr[1:] if x > pivot]

    # Рекурсивная сортировка подсписков
    less_sorted, less_calls = quick_sort(less_than_pivot)
    greater_sorted, greater_calls = quick_sort(greater_than_pivot)

    # Объединение результатов
    result = less_sorted + equal_to_pivot + greater_sorted
    total_calls = 1 + less_calls + greater_calls

    return result, total_calls


def compare_sorting_algorithms(arr):
    """
    Сравнение производительности различных алгоритмов сортировки.

    Args:
        arr (list): Список для сортировки.

    Returns:
        dict: Результаты сортировки для каждого алгоритма.
    """
    original_arr = arr.copy()
    results = {}

    algorithms = [
        ("Пузырьковая", bubble_sort),
        ("Выбором", selection_sort),
        ("Вставками", insertion_sort),
        ("Кучи", heap_sort),
        ("Слиянием", merge_sort),
        ("Быстрая", quick_sort)
    ]

    for name, sort_func in algorithms:
        test_arr = original_arr.copy()
        sorted_arr, count = sort_func(test_arr)
        results[name] = {
            "sorted": sorted_arr,
            "count": count,
            "is_sorted": sorted_arr == sorted(original_arr)
        }

    return results


def main():
    """Основная функция для демонстрации работы алгоритмов сортировки."""
    print("Введите числа через пробел для сортировки:")
    try:
        user_input = input().strip()
        if not user_input:
            print("Используем тестовый массив по умолчанию...")
            test_array = [64, 34, 25, 12, 22, 11, 90, 5]
        else:
            test_array = list(map(int, user_input.split()))

        print(f"\nИсходный массив: {test_array}")
        print(f"Длина массива: {len(test_array)}")
        print("=" * 60)

        # Сравнение всех алгоритмов
        results = compare_sorting_algorithms(test_array)

        for algo_name, result in results.items():
            print(f"\n{algo_name} сортировка:")
            print(f"  Количество операций: {result['count']}")
            print(f"  Корректность: {'✓' if result['is_sorted'] else '✗'}")
            if len(test_array) <= 10:
                print(f"  Результат: {result['sorted']}")

        # Отображение победителей по разным критериям
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")

        # Минимальное количество операций
        min_ops_algo = min(results.items(), key=lambda x: x[1]['count'])
        print(f"Меньше всего операций: {min_ops_algo[0]} ({min_ops_algo[1]['count']})")

        # Все алгоритмы корректно отсортировали
        all_correct = all(result['is_sorted'] for result in results.values())
        print(f"Все алгоритмы корректны: {'Да' if all_correct else 'Нет'}")

    except ValueError:
        print("Ошибка: пожалуйста, вводите только целые числа через пробел.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()