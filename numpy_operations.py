import numpy as np

# 1. Arithmetic Operations
print("1. Arithmetic Operations")
print("NumPy allows for element-wise arithmetic operations between arrays of the same shape or between an array and a scalar.")
print("These operations are significantly faster than performing the same operations using Python loops due to NumPy's optimized C implementations.")

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")

# Element-wise addition
addition_result = arr1 + arr2
print(f"Element-wise addition (arr1 + arr2): {addition_result}")

# Element-wise subtraction
subtraction_result = arr1 - arr2
print(f"Element-wise subtraction (arr1 - arr2): {subtraction_result}")

# Element-wise multiplication
multiplication_result = arr1 * arr2
print(f"Element-wise multiplication (arr1 * arr2): {multiplication_result}")

# Element-wise division
division_result = arr1 / arr2
print(f"Element-wise division (arr1 / arr2): {division_result}")

# Arithmetic operations with a scalar
scalar = 10
scalar_addition = arr1 + scalar
print(f"Addition with scalar ({scalar}): {scalar_addition}")
print("-" * 30)

# 2. Broadcasting
print("2. Broadcasting")
print("Broadcasting is a mechanism in NumPy that allows arithmetic operations between arrays with different shapes, provided they meet certain compatibility rules.")
print("When performing an operation between arrays of different shapes, NumPy attempts to 'broadcast' the smaller array across the larger array so that they have compatible shapes.")
print("The broadcasting rules are:")
print("  - If the arrays do not have the same number of dimensions, the shape of the smaller array is padded with ones on its left side.")
print("  - If the shapes of the two arrays do not match in any dimension, and neither dimension has a size of 1, an error is raised.")
print("  - If the shapes do not match in any dimension, and one of the dimensions has a size of 1, the array with size 1 is stretched to match the other shape.")

arr_a = np.array([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)
arr_b = np.array([10, 20, 30])          # Shape (3,)

print(f"Array A (Shape {arr_a.shape}):\n{arr_a}")
print(f"Array B (Shape {arr_b.shape}): {arr_b}")

# Broadcasting arr_b across arr_a
broadcast_addition = arr_a + arr_b
print(f"Broadcasting addition (arr_a + arr_b):\n{broadcast_addition}")

arr_c = np.array([[10], [20]]) # Shape (2, 1)
print(f"Array C (Shape {arr_c.shape}):\n{arr_c}")

# Broadcasting arr_c across arr_a
broadcast_multiplication = arr_a * arr_c
print(f"Broadcasting multiplication (arr_a * arr_c):\n{broadcast_multiplication}")
print("-" * 30)

# 3. Linear Algebra
print("3. Linear Algebra")
print("NumPy provides functions for basic linear algebra operations, which are essential in many scientific and data analysis tasks.")

matrix_a = np.array([[1, 2], [3, 4]]) # Shape (2, 2)
matrix_b = np.array([[5, 6], [7, 8]]) # Shape (2, 2)
vector_v = np.array([1, 0])           # Shape (2,)

print(f"Matrix A:\n{matrix_a}")
print(f"Matrix B:\n{matrix_b}")
print(f"Vector v: {vector_v}")

# Dot product
# For 1D arrays (vectors), it's the inner product.
# For 2D arrays (matrices), it's matrix multiplication.
# For mixing 1D and 2D arrays, it's matrix-vector multiplication.
dot_product_matrices = np.dot(matrix_a, matrix_b)
print(f"Dot product (Matrix A . Matrix B):\n{dot_product_matrices}")

dot_product_matrix_vector = np.dot(matrix_a, vector_v)
print(f"Dot product (Matrix A . Vector v): {dot_product_matrix_vector}")

# Matrix multiplication (using @ operator, available in Python 3.5+)
matrix_multiplication = matrix_a @ matrix_b
print(f"Matrix multiplication (Matrix A @ Matrix B):\n{matrix_multiplication}")

# Transpose of a matrix
transpose_matrix_a = matrix_a.T
print(f"Transpose of Matrix A:\n{transpose_matrix_a}")

# Determinant of a matrix
determinant_matrix_a = np.linalg.det(matrix_a)
print(f"Determinant of Matrix A: {determinant_matrix_a}")

# Inverse of a matrix
inverse_matrix_a = np.linalg.inv(matrix_a)
print(f"Inverse of Matrix A:\n{inverse_matrix_a}")
print("-" * 30)