import math
import os
import ast
import shutil
import time
import io
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from compiler import Compiler, PackingStrategy
from contextlib import redirect_stdout
from openfhe import *
from compiler import Operation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = TESTS_DIR / "results"
BUILD_DIR = PROJECT_ROOT / "build"
COMPUTATION_DIR = BUILD_DIR / "computation"
DISPLAY_DIR = BUILD_DIR / "display"


def reset_cache() -> None:
    ClearEvalMultKeys()
    parameters = CCParamsBFVRNS()
    parameters.SetPlaintextModulus(2**16 + 1)
    GenCryptoContext(parameters).ClearEvalAutomorphismKeys()
    ReleaseAllContexts()


def run_test(test: Path, strategy: PackingStrategy,
             remove_preprocessing: bool = True,
             runs: int = 1, warmup: int = 0,
             min_runs: int = None, time_limit: int = None) -> float:
    logging.info("=====================================================")
    logging.info(f"Testing {test}")
    logging.info("=====================================================")

    # Import source file
    source_code = test.read_text()

    # Execute in source language
    logging.info("Execute Source Code")
    logging.info("------------------------------------")
    with redirect_stdout(io.StringIO()) as f:
        start = time.perf_counter()
        exec(source_code)
        end = time.perf_counter()
        logging.info(f.getvalue().strip())
    logging.info(f"Source Code Execution Time: {end - start:.4f} seconds")

    # Compile to target language
    logging.info("------------------------------------")
    logging.info("Compile Source Code")
    logging.info("------------------------------------")
    source_ast = ast.parse(source_code)
    compiler = Compiler()
    compiler.compile(source_ast, strategy, remove_preprocessing)

    # Execute computation
    cwd = os.getcwd()
    try:
        os.chdir(COMPUTATION_DIR)
        computation_file = COMPUTATION_DIR / "computation.py"
        computation_code = computation_file.read_text()
        logging.info("------------------------------------")
        logging.info("Execute Computation")
        logging.info("------------------------------------")
        for _ in range(warmup):
            exec(computation_code)
            reset_cache()
        runtime = 0.0
        for run in range(1, runs + 1):
            start = time.perf_counter()
            exec(computation_code)
            end = time.perf_counter()
            runtime += end - start
            reset_cache()
            if (time_limit and min_runs
                    and runtime > time_limit and run >= min_runs):
                break
        runtime = runtime / runs
        logging.info(
            f"Average Computation Execution Time: {runtime:.4f} seconds")
    finally:
        os.chdir(cwd)

    # Copy results to display directory
    shutil.copytree(COMPUTATION_DIR / "results",
                    DISPLAY_DIR / "results", dirs_exist_ok=True)

    # Display output
    cwd = os.getcwd()
    try:
        os.chdir(DISPLAY_DIR)
        display_file = DISPLAY_DIR / "display.py"
        display_code = display_file.read_text()
        logging.info("------------------------------------")
        logging.info("Display Results")
        logging.info("------------------------------------")
        with redirect_stdout(io.StringIO()) as f:
            exec(display_code)
            logging.info(f.getvalue().strip())
    finally:
        os.chdir(cwd)

    return runtime


def generate_test_depth_size(op: Operation, depth: int, size: int,
                             amount: int, directory: Path) -> None:
    assert amount >= depth

    code = ("from interpreter import *\n"
            f"p = Permutation({list(range(1, size + 1))})\n")
    if op == Operation.APPLY:
        code += f"v = {list(range(1, size + 1))}\n"
    if op == Operation.INV and depth != 1:
        raise ValueError("Depth for inversion number must be 1")

    i = 0
    current_depth = depth
    for _ in range(amount):
        if current_depth >= depth:
            current_depth = 0
            i += 1
            if op == Operation.COMPOSE or op == Operation.INV:
                code += f"p{i} = p\n"
            elif op == Operation.APPLY:
                code += f"v{i} = v\n"
            else:
                raise ValueError(f"Unkown operation: {op}")
        if op == Operation.COMPOSE:
            code += f"p{i} = p{i} * p{i}\n"
        elif op == Operation.APPLY:
            code += f"v{i} = p(v{i})\n"
        elif op == Operation.INV:
            code += f"x = inv(p{i})"
        else:
            raise ValueError(f"Unkown operation: {op}")
        current_depth += 1
    file = directory / f"depth{depth}_size{size}.py"
    file.write_text(code)


def test_op_size(op: Operation, depth: int, sizes: list[int],
                 runs: int, warmup: int) -> None:
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    # Generate tests
    new_dir = TESTS_DIR / f"{op}_depth_size"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir()
    for size in sizes:
        generate_test_depth_size(op, depth, size, depth, new_dir)

    # Measure runtimes
    runtimes_diag = np.zeros(len(sizes))
    runtimes_row = np.zeros(len(sizes))
    runtimes_static = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        print(f"Measuring depth={depth}, size={size}")
        file = new_dir / f"depth{depth}_size{size}.py"
        runtimes_diag[i] = run_test(
            file, PackingStrategy.DIAG, False, runs, warmup)
        print(f"Diag: {runtimes_diag[i]}")
        runtimes_row[i] = run_test(
            file, PackingStrategy.ROW, False, runs, warmup)
        print(f"Row: {runtimes_row[i]}")
        runtimes_static[i] = run_test(
            file, PackingStrategy.CUSTOM, False, runs, warmup)
        print(f"Static: {runtimes_static[i]}")

    # Store runtimes
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / f"{op}_size_DIAG_{depth}.npy", runtimes_diag)
    np.save(RESULTS_DIR / f"{op}_size_ROW_{depth}.npy", runtimes_row)
    np.save(RESULTS_DIR / f"{op}_size_STATIC_{depth}.npy", runtimes_static)

    logging.getLogger().setLevel(old_level)


def plot_op_size(op: Operation, depth: int, sizes: list[int]) -> None:
    # Load runtimes
    runtimes_diag = np.load(RESULTS_DIR / f"{op}_size_DIAG_{depth}.npy")
    runtimes_row = np.load(RESULTS_DIR / f"{op}_size_ROW_{depth}.npy")
    runtimes_static = np.load(RESULTS_DIR / f"{op}_size_STATIC_{depth}.npy")

    # Plot runtimes
    fig = plt.figure()
    plt.plot(sizes, runtimes_diag, marker="o",
             label=r"\texttt{DIAG}", color="C0")
    plt.plot(sizes, runtimes_row, marker="s",
             label=r"\texttt{ROW}", color="C2")
    plt.plot(sizes, runtimes_static, marker="^",
             label=r"\texttt{AUTO}", color="C1", linestyle="--", dashes=(3, 3))
    plt.xlabel("Permutation Size")
    plt.ylabel("Average Runtime (s)")
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(handlelength=2.1)
    plt.grid(True)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / f"runtimes-{op}-size.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()


def test_op_depth(op: Operation, depths: list[int], size: int,
                  runs: int, warmup: int) -> None:
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    # Generate tests
    new_dir = TESTS_DIR / f"{op}_depth_size"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir()
    for depth in depths:
        generate_test_depth_size(op, depth, size, depth, new_dir)

    # Measure runtimes
    runtimes_diag = np.zeros(len(depths))
    runtimes_row = np.zeros(len(depths))
    runtimes_static = np.zeros(len(depths))
    for i, depth in enumerate(depths):
        print(f"Measuring depth={depth}, size={size}")
        file = new_dir / f"depth{depth}_size{size}.py"
        runtimes_diag[i] = run_test(
            file, PackingStrategy.DIAG, False, runs, warmup)
        print(f"Diag: {runtimes_diag[i]}")
        runtimes_row[i] = run_test(
            file, PackingStrategy.ROW, False, runs, warmup)
        print(f"Row: {runtimes_row[i]}")
        runtimes_static[i] = run_test(
            file, PackingStrategy.CUSTOM, False, runs, warmup)
        print(f"Static: {runtimes_static[i]}")

    # Store runtimes
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / f"{op}_depth_DIAG_{size}.npy", runtimes_diag)
    np.save(RESULTS_DIR / f"{op}_depth_ROW_{size}.npy", runtimes_row)
    np.save(RESULTS_DIR / f"{op}_depth_STATIC_{size}.npy", runtimes_static)

    logging.getLogger().setLevel(old_level)


def plot_op_depth(op: Operation, depths: list[int], size: int) -> None:
    # Load runtimes
    runtimes_diag = np.load(RESULTS_DIR / f"{op}_depth_DIAG_{size}.npy")
    runtimes_row = np.load(RESULTS_DIR / f"{op}_depth_ROW_{size}.npy")
    runtimes_static = np.load(RESULTS_DIR / f"{op}_depth_STATIC_{size}.npy")

    # Plot runtimes
    fig = plt.figure()
    plt.plot(depths, runtimes_diag, marker="o",
             label=r"\texttt{DIAG}", color="C0")
    plt.plot(depths, runtimes_row, marker="s",
             label=r"\texttt{ROW}", color="C2")
    plt.plot(depths, runtimes_static, marker="^",
             label=r"\texttt{AUTO}", color="C1", linestyle="--", dashes=(3, 3))
    plt.xlabel("Nesting Level")
    plt.ylabel("Average Runtime (s)")
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(handlelength=2.1)
    plt.grid(True)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / f"runtimes-{op}-depth.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()


def generate_test_amount_size(op: Operation, amount: int, size: int,
                              directory: Path) -> None:
    code = ("from interpreter import *\n"
            f"p = Permutation({list(range(1, size + 1))})\n")
    if op == Operation.APPLY:
        code += f"v = {list(range(1, size + 1))}\n"

    for _ in range(amount):
        if op == Operation.COMPOSE:
            code += f"x = p * p\n"
        elif op == Operation.APPLY:
            code += f"x = p(v)\n"
        else:
            raise ValueError(f"Unkown operation: {op}")

    file = directory / f"amount{amount}_size{size}.py"
    file.write_text(code)


def test_preprocessing_amount(op: Operation, strategy: PackingStrategy,
                              amounts: list[int], size: int, runs: int,
                              warmup: int) -> None:
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    # Generate tests
    new_dir = TESTS_DIR / f"{op}_amount_size"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir()
    for amount in amounts:
        generate_test_amount_size(op, amount, size, new_dir)

    # Measure runtimes
    t_original = np.zeros(len(amounts))
    t_optimized = np.zeros(len(amounts))
    for i, amount in enumerate(amounts):
        print(f"Strategy={strategy}, op={op}, amount={amount}")
        file = new_dir / f"amount{amount}_size{size}.py"
        runtimes_original = []
        runtimes_optimized = []
        for _ in range(warmup):
            run_test(file, strategy, False, 5, 1)
            run_test(file, strategy, True, 5, 1)
        for run in range(runs):
            runtimes_optimized.append(run_test(file, strategy, True, 20, 5))
            runtimes_original.append(run_test(file, strategy, False, 20, 5))
            print(f"Ratio={runtimes_optimized[run] / runtimes_original[run]}")
        t_original[i] = sum(runtimes_original) / runs
        t_optimized[i] = sum(runtimes_optimized) / runs
        print(t_optimized[i] / t_original[i])

    # Store runtimes
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR /
            f"preproc_included_{op}_{strategy}_{size}.npy", t_original)
    np.save(RESULTS_DIR /
            f"preproc_removed_{op}_{strategy}_{size}.npy", t_optimized)

    logging.getLogger().setLevel(old_level)


def plot_preprocessing_amount(amounts: list[int], size: int) -> None:
    # Load runtimes
    t_original = np.zeros((3, len(amounts)))
    t_optimized = np.zeros((3, len(amounts)))
    for i, (strategy, op) in enumerate([
        (PackingStrategy.DIAG, Operation.APPLY),
        (PackingStrategy.DIAG, Operation.COMPOSE),
        (PackingStrategy.ROW, Operation.COMPOSE)
    ]):
        t_original[i] = np.load(
            RESULTS_DIR / f"preproc_included_{op}_{strategy}_{size}.npy")
        t_optimized[i] = np.load(
            RESULTS_DIR / f"preproc_removed_{op}_{strategy}_{size}.npy")

    # Plot runtimes
    plt.figure()
    plt.plot(amounts, t_optimized[0] / t_original[0],
             marker="o", label=r"\texttt{DIAG}, \texttt{p(v)}")
    plt.plot(amounts, t_optimized[1] / t_original[1],
             marker="s", label=r"\texttt{DIAG}, \texttt{p * p}")
    plt.plot(amounts, t_optimized[2] / t_original[2],
             marker="^", label=r"\texttt{ROW}, \texttt{p(v)} / \texttt{p * p}")
    plt.xlabel("Number of Operations")
    plt.ylabel(r"$t_{\mathrm{opt}} / t_{\mathrm{orig}}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "runtime-ratios-preprocessing-amount.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()


def test_preprocessing_size(op: Operation, strategy: PackingStrategy,
                            amount: int, sizes: list[int], runs: int,
                            warmup: int) -> None:
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    # Generate tests
    new_dir = TESTS_DIR / f"{op}_amount_size"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir()
    for size in sizes:
        generate_test_amount_size(op, amount, size, new_dir)

    # Measure runtimes
    t_original = np.zeros(len(sizes))
    t_optimized = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        print(f"Strategy={strategy}, op={op}, size={size}")
        file = new_dir / f"amount{amount}_size{size}.py"
        runtimes_original = []
        runtimes_optimized = []
        for _ in range(warmup):
            run_test(file, strategy, True, 5, 1)
            run_test(file, strategy, False, 5, 1)
        for run in range(runs):
            runtimes_optimized.append(run_test(file, strategy, True, 20, 5))
            runtimes_original.append(run_test(file, strategy, False, 20, 5))
            print(f"Ratio={runtimes_optimized[run] / runtimes_original[run]}")
        t_original[i] = sum(runtimes_original) / runs
        t_optimized[i] = sum(runtimes_optimized) / runs
        print(t_optimized[i] / t_original[i])

    # Store runtimes
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR /
            f"preproc_included_{op}_{strategy}_amount{amount}.npy", t_original)
    np.save(RESULTS_DIR /
            f"preproc_removed_{op}_{strategy}_amount{amount}.npy", t_optimized)

    logging.getLogger().setLevel(old_level)


def plot_preprocessing_size(sizes: list[int], amount: int) -> None:
    # Load runtimes
    t_original = np.zeros((3, len(sizes)))
    t_optimized = np.zeros((3, len(sizes)))
    for i, (strategy, op) in enumerate([
        (PackingStrategy.DIAG, Operation.APPLY),
        (PackingStrategy.DIAG, Operation.COMPOSE),
        (PackingStrategy.ROW, Operation.COMPOSE)
    ]):
        t_original[i] = np.load(
            RESULTS_DIR / f"preproc_included_{op}_{strategy}_amount{amount}.npy")
        t_optimized[i] = np.load(
            RESULTS_DIR / f"preproc_removed_{op}_{strategy}_amount{amount}.npy")

    # Plot runtimes
    plt.figure()
    plt.plot(sizes, t_optimized[0] / t_original[0],
             marker="o", label=r"\texttt{DIAG}, \texttt{p(v)}")
    plt.plot(sizes, t_optimized[1] / t_original[1],
             marker="s", label=r"\texttt{DIAG}, \texttt{p * p}")
    plt.plot(sizes, t_optimized[2] / t_original[2],
             marker="^", label=r"\texttt{ROW}, \texttt{p(v)} / \texttt{p * p}")
    plt.xlabel("Permutation Size")
    plt.ylabel(r"$t_{\mathrm{opt}} / t_{\mathrm{orig}}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "runtime-ratios-preprocessing-size.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()


def plot_rotation_indcies(sizes: list[int]) -> None:
    indices_diag_vec = sizes
    indices_diag_inv = []
    indices_row_vec = []
    indices_row_inv = []
    for size in sizes:
        padded_size = 2**math.ceil(math.log2(size))
        rotations_diag_inv = set()
        rotations_row_vec = set()
        rotations_row_inv = set()

        # DIAG, inv(p)
        rotations_diag_inv.update(
            i * (1 - padded_size) - j for j in range(size)
            for i in range(size) if not (i == j == 0)
        )
        rotations_diag_inv.update(
            2**i for i in range(int(math.log2(padded_size**2)))
        )
        indices_diag_inv.append(len(rotations_diag_inv))

        # ROW, p(v) / p * p
        for i in range(padded_size):
            rotations_row_vec.update({
                i,
                i * padded_size
            })
        for i in range(int(math.log2(padded_size))):
            rotations_row_vec.update({
                -2**i,
                -2**i * padded_size,
            })
        indices_row_vec.append(len(rotations_row_vec))

        # ROW, inv(p)
        rotations_row_inv.update(
            2**i for i in range(int(math.log2(padded_size**2)))
        )
        indices_row_inv.append(len(rotations_row_inv))

    fig = plt.figure()
    plt.plot(sizes, indices_diag_vec, marker="o", color="C0",
             label=r"\texttt{DIAG}")
    plt.plot(sizes, indices_diag_inv, marker="o", color="C1",
             label=r"\texttt{DIAG}, \texttt{inv(p)}")
    plt.plot(sizes, indices_row_vec, marker="s", color="C2",
             label=r"\texttt{ROW}")
    plt.plot(sizes, indices_row_inv, marker="s", color="C3",
             label=r"\texttt{ROW}, \texttt{inv(p)}")
    plt.xlabel("Permutation Size")
    plt.ylabel(r"Number of Rotation Indices")
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "rotation-indices.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 15,
        'text.usetex': True,
    })

    run_test(TESTS_DIR / "example.py", PackingStrategy.CUSTOM)
