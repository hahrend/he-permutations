import math
from enum import StrEnum
from dataclasses import dataclass


class DataType(StrEnum):
    INT = "DataType.INT"
    LIST = "DataType.LIST"
    PERMUTATION = "DataType.PERMUTATION"


class Layout(StrEnum):
    ROW = "Layout.ROW"
    DIAG = "Layout.DIAG"


@dataclass
class Packing:
    data_type: DataType
    layout: Layout
    size: int
    padded_size: int

    def __repr__(self):
        return (f"Packing(data_type={self.data_type}, "
                f"layout={self.layout.value if self.layout else None}, "
                f"size={self.size if self.size else None}, "
                f"padded_size={self.padded_size if self.padded_size else None})")


def to_matrix(mapping: list[int]) -> list[list[int]]:
    """Create a permutation matrix for the mapping."""
    n = len(mapping)
    matrix = [[0] * n for _ in range(n)]
    for i, p in enumerate(mapping):
        matrix[p-1][i] = 1
    return matrix


def to_mapping(matrix: list[list[int]]) -> list[int]:
    """Create a mapping for the permutation matrix."""
    return [column.index(1) + 1 for column in zip(*matrix)]


def pad_array_to_matrix(array: list[int]) -> list[list[int]]:
    """Pad the array with zeros to become a square matrix.

    The array is interpreted as a column vector, which is extended by adding
    zero columns to the right.

    Returns:
        The rows of the resulting matrix.
    """
    dim = len(array)
    return [
        [array[i]] + [0] * (dim - 1) for i in range(dim)
    ]


def unpad_array(matrix: list[list[int]]) -> list[int]:
    """Remove the padding from an array padded with `pad_array_to_matrix`."""
    return [row[0] for row in matrix]


def pad_matrix_to_power_of_2(matrix: list[list[int]]) -> list[list[int]]:
    """Pad the square matrix with zeros until its dimension divides the next power of 2."""
    dim = len(matrix)
    padded_dim = 2**math.ceil(math.log2(dim))  # Next power of 2

    padded_matrix = [row + [0] * (padded_dim - dim) for row in matrix]
    padded_matrix.extend([[0] * padded_dim for _ in range(padded_dim - dim)])
    return padded_matrix


def pack_matrix_row(matrix: list[list[int]]) -> tuple[list[int], int]:
    """Pack the matrix in row-major order.

    The (square) matrix is padded with zero entries to fit a power of 2 and
    encoded as the concatenation of the resulting rows.

    Returns:
        A tuple consisting of the packed matrix and the padded dimension.
    """
    padded_matrix = pad_matrix_to_power_of_2(matrix)
    encoding = [element for row in padded_matrix for element in row]
    return encoding, len(padded_matrix)


def unpack_matrix_row(encoding: list[int], dim: int,
                      padded_dim: int) -> list[list[int]]:
    """Unpack a row-major packed matrix.

    The concatenated rows of the matrix are seperated from each other. A padding
    is removed by extracting the resulting upper-left submatrix. 

    Returns:
        The rows of the matrix with size (dim,dim).
    """
    padded_matrix = [encoding[i*padded_dim:(i+1)*padded_dim]
                     for i in range(padded_dim)]

    return [padded_matrix[i][:dim] for i in range(dim)]


def pack_matrix_diag(matrix: list[list[int]]) -> list[list[int]]:
    """Pack the matrix with the Halevi-Shoup method.

    The (square) matrix is encoded as an array of its upper diagonals.

    Returns:
        The pack matrix.
    """
    dim = len(matrix)
    return [[matrix[j][(i + j) % dim] for j in range(dim)] for i in range(dim)]


def unpack_matrix_diag(diagonals: list[list[int]]) -> list[list[int]]:
    """Unpack a Halevi-Shoup packed matrix.

    The diagonal packing is transformed to a row encoding.

    Returns:
        The rows of the matrix.
    """
    dim = len(diagonals)
    return [[diagonals[j - i][i] for j in range(dim)] for i in range(dim)]
