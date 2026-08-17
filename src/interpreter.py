from __future__ import annotations
from openfhe import *
from utils import *

SERTYPE = SERBINARY | SERJSON


# ------------------------------ Source Language ------------------------------

class Permutation:

    def __init__(self, mapping: list[int]) -> None:
        self.n = len(mapping)
        if (not mapping or min(mapping) != 1
                or max(mapping) != self.n or len(set(mapping)) != self.n):
            raise ValueError("Mapping does not define a valid permutation")
        self.mapping = mapping

    def __call__(self, x: list[int]) -> list[int]:
        if len(x) != self.n:
            raise ValueError("Permutation length does not match")
        result = [None] * len(x)
        for i, p in enumerate(self.mapping):
            result[p - 1] = x[i]

        return result

    def __mul__(self, right: Permutation) -> Permutation:
        if not isinstance(right, Permutation):
            raise TypeError("Can only multiply with permutations")
        if self.n != len(right.mapping):
            raise ValueError("Permuatations must have the same length")
        new_mapping = [self.mapping[i - 1] for i in right.mapping]

        return Permutation(new_mapping)

    def __str__(self) -> str:
        max_width = len(str(self.n))
        domain = " ".join(
            f"{i+1:>{max_width}}" for i in range(self.n))
        image = " ".join(f"{j:>{max_width}}" for j in self.mapping)

        return f"[{domain}\n {image}]"


def inv(p: Permutation) -> int:
    if not isinstance(p, Permutation):
        raise TypeError("Argument is not a permutation")
    count = 0
    for i in range(p.n):
        for j in range(i + 1, p.n):
            if p.mapping[i] > p.mapping[j]:
                count += 1
    return count


# ----------------------- Target Language (Computation) -----------------------

def preprocess_left_matrix_row(cc: CryptoContext, A: Ciphertext,
                               dim: int) -> list[Ciphertext]:
    """Preprocess a row-major encoded matrix for matrix-multipliacation.

    It should be the left multiplicand and a square matrix of size dim^2,
    where dim is a power of 2"""
    A_hats = []
    for i in range(dim):
        pi_mask = cc.MakePackedPlaintext(
            [1 if i == j % dim else 0 for j in range(dim**2)]
        )
        if i == 0:
            A_hat = cc.EvalMult(A, pi_mask)
        else:
            A_hat = cc.EvalRotate(cc.EvalMult(A, pi_mask), i)
        for j in range(int(math.log2(dim))):
            A_hat = cc.EvalAdd(A_hat, cc.EvalRotate(A_hat, -2**j))
        A_hats.append(A_hat)
    return A_hats


def preprocess_right_matrix_row(cc: CryptoContext, B: Ciphertext,
                                dim: int) -> list[Ciphertext]:
    """Preprocess a row-major encoded matrix for matrix-multipliacation.

    It should be the right multiplicand and a square matrix of size dim^2,
    where dim is a power of 2"""
    B_hats = []
    for i in range(dim):
        psi_mask = cc.MakePackedPlaintext(
            [1 if i * dim <= j < (i + 1) * dim else 0 for j in range(dim**2)]
        )
        if i == 0:
            B_hat = cc.EvalMult(B, psi_mask)
        else:
            B_hat = cc.EvalRotate(cc.EvalMult(B, psi_mask), i * dim)
        for j in range(int(math.log2(dim))):
            B_hat = cc.EvalAdd(B_hat, cc.EvalRotate(B_hat, -2**j * dim))
        B_hats.append(B_hat)
    return B_hats


def matrix_matrix_mult_row(cc: CryptoContext, A_hats: list[Ciphertext],
                           B_hats: list[Ciphertext], dim: int) -> Ciphertext:
    """Compute the matrix product A * B for two preprocessed matrices."""
    mults = [
        cc.EvalMult(A_hats[i], B_hats[i])
        for i in range(dim)
    ]
    return cc.EvalAddMany(mults)


def preprocess_vector_diag(cc: CryptoContext, vector: Ciphertext,
                           dim: int) -> Ciphertext:
    """Repeat vector entries once to make rotations cyclic."""
    return cc.EvalAdd(vector, cc.EvalRotate(vector, -dim))


def matrix_vector_mult_diag(cc: CryptoContext, matrix: list[Ciphertext],
                            vector: Ciphertext) -> Ciphertext:
    """Multiply diagonal encoded matrix with a preprocessed vector."""
    n = len(matrix)

    mults = []
    for i in range(n):
        if i == 0:
            mults.append(cc.EvalMult(matrix[i], vector))
        else:
            mults.append(cc.EvalMult(matrix[i], cc.EvalRotate(vector, i)))

    return cc.EvalAddMany(mults)


def preprocess_matrix_diag(cc: CryptoContext,
                           B: list[Ciphertext]) -> list[Ciphertext]:
    """Repeat each diagonal of B once to make rotations cyclic."""
    return [cc.EvalAdd(diag, cc.EvalRotate(diag, -len(B))) for diag in B]


def matrix_matrix_mult_diag(cc: CryptoContext, A: list[Ciphertext],
                            B: list[Ciphertext]) -> Ciphertext:
    """Compute the matrix product A * B for two diagonal encoded matrices.

    The matrix B must have been preprocessed before to make rotations cyclic."""
    n = len(A)

    result = []
    for i in range(n):
        mults = []
        for j in range(n):
            if j == 0:
                mults.append(cc.EvalMult(A[j], B[i-j]))
            else:
                mults.append(cc.EvalMult(A[j], cc.EvalRotate(B[i-j], j)))
        result.append(cc.EvalAddMany(mults))
    return result


def inversion_number_row(cc: CryptoContext, P: Ciphertext,
                         original_dim: int, dim: int) -> Ciphertext:
    """Compute the inversion number for a row-major packed permutation matrix P."""
    mask = cc.MakePackedPlaintext([0])
    for i in range(original_dim - 1):
        # Extract i-th column
        pi_mask = cc.MakePackedPlaintext(
            [1 if i == j % dim else 0 for j in range(dim**2)]
        )
        column = cc.EvalMult(P, pi_mask)

        # Create i-th mask
        for j in range(int(math.log2(dim**2))):
            column = cc.EvalAdd(column, cc.EvalRotate(column, 2**j))
        right_mask = cc.MakePackedPlaintext(
            [1 if i < j % dim else 0 for j in range(dim**2)]
        )
        mask = cc.EvalAdd(mask, (cc.EvalMult(column, right_mask)))

    # Sum inversions
    result = cc.EvalMult(P, mask)
    for i in range(int(math.log2(dim**2))):
        result = cc.EvalAdd(result, cc.EvalRotate(result, 2**i))
    return result


def convert_diag_to_row(cc: CryptoContext, A: list[Ciphertext]) -> Ciphertext:
    """Change the packing of matrix A from diagonal to row-major."""
    dim = len(A)
    padded_dim = 2**math.ceil(math.log2(dim))  # Next power of 2

    result = cc.MakePackedPlaintext([0])
    for i in range(dim):
        mask = cc.MakePackedPlaintext(
            [1 if i == j else 0 for j in range(dim)]
        )
        for j in range(dim):
            element = cc.EvalMult(A[j - i], mask)
            if i == j == 0:
                result = cc.EvalAdd(result, element)
            else:
                result = cc.EvalAdd(result, cc.EvalRotate(
                    element, i * (1 - padded_dim) - j))
    return result


def serialize(file_name: str, ciphertext: Ciphertext,
              serial_type: SERTYPE) -> None:
    if not SerializeToFile(file_name, ciphertext, serial_type):
        raise IOError(
            f"Error writing serialization of {ciphertext} to {file_name}")


def serialize_many(file_names: str, ciphertexts: list[Ciphertext],
                   serial_type: SERTYPE) -> None:
    for file_name, ciphertext in zip(file_names, ciphertexts):
        serialize(file_name, ciphertext, serial_type)


def deserialize(file_name: str, serial_type: SERTYPE) -> Ciphertext:
    ciphertext, result = DeserializeCiphertext(file_name, serial_type)
    if not result:
        raise IOError(f"Error reading from file {file_name}")
    return ciphertext


def deserialize_many(file_names: list[str],
                     serial_type: SERTYPE) -> list[Ciphertext]:
    return [deserialize(file, serial_type) for file in file_names]


def deserialize_crypto_context(file_name: str,
                               serial_type: SERTYPE) -> CryptoContext:
    cc, result = DeserializeCryptoContext(file_name, serial_type)
    if not result:
        raise IOError(f"Error reading from file {file_name}")
    return cc


def deserialize_public_key(file_name: str,
                           serial_type: SERTYPE) -> PublicKey:
    public_key, result = DeserializePublicKey(file_name, serial_type)
    if not result:
        raise IOError(f"Error reading from file {file_name}")
    return public_key


def deserialize_private_key(file_name: str,
                            serial_type: SERTYPE) -> PrivateKey:
    private_key, result = DeserializePrivateKey(file_name, serial_type)
    if not result:
        raise IOError(f"Error reading from file {file_name}")
    return private_key


def deserialize_mult_key(file_name: str,
                         serial_type: SERTYPE, cc: CryptoContext) -> None:
    if not cc.DeserializeEvalMultKey(file_name, serial_type):
        raise IOError(f"Error reading from file {file_name}")


def deserialize_rotation_key(file_name: str,
                             serial_type: SERTYPE, cc: CryptoContext) -> None:
    if not cc.DeserializeEvalAutomorphismKey(file_name, serial_type):
        raise IOError(f"Error reading from file {file_name}")


# ------------------------- Target Language (Output) --------------------------


def print_ciphertext(file_name: str, packing: Packing, serial_type: SERTYPE,
                     cc: CryptoContext, private_key: PrivateKey) -> None:
    match packing:
        case Packing(data_type=DataType.INT):
            plaintext = decrypt(file_name, 1, serial_type, cc, private_key)
            print(plaintext[0])
        case Packing(data_type=DataType.LIST, layout=Layout.ROW):
            plaintext = decrypt(
                file_name, packing.padded_size**2, serial_type, cc, private_key)
            matrix = unpack_matrix_row(
                plaintext, packing.size, packing.padded_size)
            print(unpad_array(matrix))
        case Packing(data_type=DataType.PERMUTATION, layout=Layout.ROW):
            plaintext = decrypt(
                file_name, packing.padded_size**2, serial_type, cc, private_key)
            matrix = unpack_matrix_row(
                plaintext, packing.size, packing.padded_size)
            print(Permutation(to_mapping(matrix)))
        case Packing(data_type=DataType.LIST, layout=Layout.DIAG):
            plaintext = decrypt(
                file_name, packing.size, serial_type, cc, private_key)
            print(plaintext)
        case _:
            raise ValueError(f"Unsupported packing {packing}")


def print_many(file_names: list[str], packing: Packing, serial_type: SERTYPE,
               cc: CryptoContext, private_key: PrivateKey) -> None:
    match packing:
        case Packing(data_type=DataType.PERMUTATION, layout=Layout.DIAG):
            plaintext_diagonals = decrypt_many(
                file_names, [packing.padded_size] * len(file_names),
                serial_type, cc, private_key)
            matrix = unpack_matrix_diag(plaintext_diagonals)
            print(Permutation(to_mapping(matrix)))
        case _:
            raise ValueError(f"Unsupported packing {packing}")


def decrypt(file_name: str, length: int, serial_type: SERTYPE,
            cc: CryptoContext, private_key: PrivateKey) -> list[int]:
    ciphertext = deserialize(file_name, serial_type)
    plaintext = cc.Decrypt(private_key, ciphertext)
    plaintext.SetLength(length)
    return plaintext.GetPackedValue()


def decrypt_many(file_names: list[str], lengths: list[int], serial_type: SERTYPE,
                 cc: CryptoContext, private_key: PrivateKey) -> list[list[int]]:
    return [decrypt(file_name, length, serial_type, cc, private_key)
            for file_name, length in zip(file_names, lengths)]
