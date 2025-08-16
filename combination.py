import pandas as pd
import numpy as np

# 1. Relationship between Pandas and NumPy
print("1. Relationship between Pandas and NumPy")
print("Pandas is built on top of NumPy. Its core data structures, Series and DataFrame, fundamentally rely on NumPy arrays.")
print("A Pandas Series is essentially a one-dimensional NumPy array with an index.")
print("A Pandas DataFrame is a collection of Series objects, where each Series represents a column, all sharing the same index (for rows).")
print("This means that many operations in Pandas are vectorized and efficient because they are ultimately performed on the underlying NumPy arrays.")
print("-" * 30)

# 2. Scenarios where combining NumPy and Pandas is beneficial
print("2. Scenarios where combining NumPy and Pandas is beneficial")
print("Combining NumPy and Pandas is powerful for:")
print("  - Leveraging NumPy's mathematical and array operations")
print("  - Performance-critical operations")
print("  - Interfacing with libraries that expect NumPy arrays")
print("  - Advanced indexing and boolean operations")
print("-" * 30)

# 3. Converting between Pandas DataFrames/Series and NumPy arrays
print("3. Converting between Pandas DataFrames/Series and NumPy arrays")
print("Pandas objects can be easily converted to NumPy arrays using the `.values` or `.to_numpy()` attributes/methods.")

# Create a sample DataFrame
data = {'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]}
df = pd.DataFrame(data)
print("Original Pandas DataFrame:")
print(df)

# Convert DataFrame to NumPy array
numpy_array_from_df = df.values  # or df.to_numpy()
print(f"\nNumPy array from DataFrame using .values:\n{numpy_array_from_df}")
print(f"Type: {type(numpy_array_from_df)}")

# Create a sample Series
s = pd.Series([10, 20, 30, 40])
print("\nOriginal Pandas Series:")
print(s)

# Convert Series to NumPy array
numpy_array_from_s = s.values  # or s.to_numpy()
print(f"\nNumPy array from Series using .values:\n{numpy_array_from_s}")
print(f"Type: {type(numpy_array_from_s)}")

# Converting NumPy array to Pandas Series/DataFrame
print("\nConverting NumPy array back to Pandas Series/DataFrame:")
np_arr = np.array([[1, 2, 3], [4, 5, 6]])
df_from_np = pd.DataFrame(np_arr, columns=['A', 'B', 'C'])
print("DataFrame from NumPy array:")
print(df_from_np)

np_s_arr = np.array([100, 200, 300])
s_from_np = pd.Series(np_s_arr)
print("\nSeries from NumPy array:")
print(s_from_np)
print("-" * 30)

# 4. Demonstrating using NumPy functions on Pandas DataFrames or Series
print("4. Demonstrating using NumPy functions on Pandas DataFrames or Series")
print("Original DataFrame:")
print(df)

# Using NumPy's sqrt function on a Series
sqrt_series = np.sqrt(df['col1'])
print(f"\nApplying np.sqrt() to 'col1' Series:\n{sqrt_series}")

# Using NumPy's sin function on a Series
sin_series = np.sin(df['col2'])
print(f"\nApplying np.sin() to 'col2' Series:\n{sin_series}")

# Using NumPy's sum function on a DataFrame (sums columns by default)
sum_df = np.sum(df)
print(f"\nApplying np.sum() to DataFrame (sums columns):\n{sum_df}")

# Using NumPy's mean function on a specific axis
mean_rows = np.mean(df, axis=1)  # axis=1 for rows
print(f"\nApplying np.mean() to DataFrame along axis=1 (mean of rows):\n{mean_rows}")
print("-" * 30)

# 5. Examples of applying NumPy operations (universal functions, linear algebra) to Pandas objects
print("5. Examples of applying NumPy operations (universal functions, linear algebra) to Pandas objects")
print("Original DataFrame:")
print(df)

# Element-wise addition using NumPy ufunc
add_result = np.add(df['col1'], df['col2'])
print(f"\nApplying np.add() to 'col1' and 'col2' Series:\n{add_result}")

# Element-wise exponential using NumPy ufunc
exp_series = np.exp(df['col1'])
print(f"\nApplying np.exp() to 'col1' Series:\n{exp_series}")

# Linear Algebra (requires NumPy arrays)
matrix_a_df = pd.DataFrame([[1, 2], [3, 4]])
matrix_b_df = pd.DataFrame([[5, 6], [7, 8]])
print("\nMatrix A (DataFrame):\n", matrix_a_df)
print("\nMatrix B (DataFrame):\n", matrix_b_df)

matrix_a_np = matrix_a_df.values
matrix_b_np = matrix_b_df.values
matrix_product_np = np.dot(matrix_a_np, matrix_b_np)
matrix_product_df = pd.DataFrame(matrix_product_np)
print(f"\nMatrix product (NumPy array result):\n{matrix_product_np}")
print("\nMatrix product (converted back to DataFrame):\n")
print(matrix_product_df)
print("-" * 30)

# 6. Illustrating broadcasting rules
print("6. Illustrating how broadcasting rules from NumPy apply to Pandas operations")
print("Original DataFrame:")
print(df)

# Broadcasting scalar
scalar = 10
df_scalar_add = df + scalar
print(f"\nDataFrame + scalar ({scalar}):\n{df_scalar_add}")

# Broadcasting Series aligned by columns
series_to_broadcast_cols = pd.Series([100, 200], index=['col1', 'col2'])
print("\nSeries to broadcast (aligned with DataFrame columns):")
print(series_to_broadcast_cols)

df_series_add_cols = df + series_to_broadcast_cols
print("\nDataFrame + Series (aligned by column names):\n")
print(df_series_add_cols)

# Broadcasting Series aligned by index (rows)
series_to_broadcast_rows = pd.Series([5, 10, 15, 20], index=df.index)
print("\nSeries to broadcast (aligned with DataFrame rows):")
print(series_to_broadcast_rows)

df_series_add_rows = df.add(series_to_broadcast_rows, axis=0)
print("\nDataFrame + Series (aligned by index, row-wise):\n")
print(df_series_add_rows)

# Mismatched index
series_mismatch_index = pd.Series([1000, 2000], index=[0, 5])
print("\nSeries with mismatching index:")
print(series_mismatch_index)

df_series_add_mismatch = df + series_mismatch_index
print("\nDataFrame + Series (with mismatch - introduces NaNs):\n")
print(df_series_add_mismatch)

print("Pandas' broadcasting, combined with index alignment, makes operations intuitive for data analysis.")
print("-" * 30)
