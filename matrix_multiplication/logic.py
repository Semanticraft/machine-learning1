def matrix_multiplication(A: list, B: list) -> list:
    """
    Multiply two matrices A and B and return the resulting matrix.

    Parameters:
        A (list): Matrix A.
        B (list): Matrix B.

    Returns:
        list: A * B.
    """
    if len(A[0]) != len(B):
        raise ValueError(f"Number of columns {len(A[0])} in A does not equal the number of rows {len(B)} in B.")
    if not matrix_consists_of_numbers(A) or not matrix_consists_of_numbers(B):
        raise TypeError("Matrices to be multiplied have non-number values.")
    C = [[0] * len(A) for _ in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_consists_of_numbers(matrix: list) -> bool:
    """
    Check if matrix only consists of numbers.

    Parameters:
        matrix (list): The given matrix.

    Returns:
        bool: True, if matrix only consists of numbers, False else wise.
    """
    for row in matrix:
        for element in row:
            if not isinstance(element, (int, float)):
                return False
    return True
