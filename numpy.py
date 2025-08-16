# 1. What is NumPy?
print("1. What is NumPy?")
print("NumPy (Numerical Python) is a fundamental package for numerical computation in Python.")
print("It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.")
print("Its importance lies in providing efficient array operations, which are crucial for tasks in data science, machine learning, scientific computing, and more.")
print("-" * 30)

# 2. The ndarray
print("2. The ndarray")
print("The core data structure in NumPy is the ndarray (n-dimensional array).")
print("It is a homogeneous container for fixed-size items. This means all elements in an ndarray must be of the same data type.")
print("This homogeneity and fixed size are key to NumPy's performance compared to Python's built-in lists.")
print("-" * 30)

# 3. NumPy Data Types (dtypes)
print("3. NumPy Data Types (dtypes)")
print("NumPy arrays have a specific data type (dtype) that describes the type of elements in the array.")
print("This ensures that all elements in the array are treated consistently.")
print("Common dtypes include:")
print("  - int: Signed integer types (e.g., int8, int16, int32, int64)")
print("  - float: Floating-point types (e.g., float16, float32, float64)")
print("  - bool: Boolean type (True or False)")
print("  - complex: Complex number types (e.g., complex64, complex128)")
print("  - object: Python object type (less common for performance reasons)")
print("-" * 30)

# 4. Creating NumPy Arrays
print("4. Creating NumPy Arrays")
print("NumPy provides various functions to create arrays:")
print("  - np.array(): Create an array from a Python list or tuple.")
print("  - np.zeros(shape): Create an array filled with zeros.")
print("  - np.ones(shape): Create an array filled with ones.")
print("  - np.empty(shape): Create an array with uninitialized (random) data.")
print("  - np.arange(start, stop, step): Create an array with evenly spaced values within a given interval.")
print("  - np.linspace(start, stop, num): Create an array with a specified number of evenly spaced values over a given interval.")
print("-" * 30)

# Example array creation
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
print(f"Example np.array(): {arr1}")
print(f"dtype of arr1: {arr1.dtype}")

arr_zeros = np.zeros((2, 3))
print(f"Example np.zeros((2, 3)): \n{arr_zeros}")

arr_ones = np.ones((3, 2))
print(f"Example np.ones((3, 2)): \n{arr_ones}")

arr_empty = np.empty((2, 2))
print(f"Example np.empty((2, 2)): \n{arr_empty}") # Values will be uninitialized

arr_arange = np.arange(0, 10, 2)
print(f"Example np.arange(0, 10, 2): {arr_arange}")

arr_linspace = np.linspace(0, 1, 5)
print(f"Example np.linspace(0, 1, 5): {arr_linspace}")
print("-" * 30)

# 5. Indexing and Slicing NumPy Arrays
print("5. Indexing and Slicing NumPy Arrays")
print("NumPy arrays can be indexed and sliced similar to Python lists, but with more powerful options.")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original array:\n{arr}")

# Basic Indexing
print(f"Basic indexing (element at row 1, column 2): {arr[1, 2]}")

# Slicing
print(f"Slicing (first row): {arr[0, :]}")
print(f"Slicing (second column): {arr[:, 1]}")
print(f"Slicing (subarray from row 0 to 1, column 1 to 2): \n{arr[0:2, 1:3]}")

# Boolean Indexing
print(f"Boolean indexing (elements greater than 5): {arr[arr > 5]}")

# Fancy Indexing
print(f"Fancy indexing (elements at specific indices): {arr[[0, 2], [1, 0]]}") # Elements at (0, 1) and (2, 0)
print("-" * 30)