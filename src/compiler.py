from ast import *
from openfhe import *
from utils import *
import interpreter
from interpreter import SERTYPE
import os
import inspect
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from itertools import count
from enum import auto
import logging

Binding = tuple[Name, expr]
Temporaries = list[Binding]


class PackingStrategy(StrEnum):
    ROW = auto()
    DIAG = auto()
    CUSTOM = auto()


class Operation(StrEnum):
    APPLY = auto()
    COMPOSE = auto()
    INV = auto()


@dataclass(frozen=True)
class Node:
    id: int
    value: Operation | Tuple


class Compiler:
    COST_BY_DEPTH: dict[int, tuple[float, float]] = {
        0: (1.1734933000298044e-05, 0.005159931681999296, 2.9871788001400994e-05, 0.0009216093650029506),
        1: (1.9811453000329492e-05, 0.007237820570999249, 5.122972300068795e-05, 0.0016447245590006786),
        2: (2.4357985000278862e-05, 0.007158756108999115, 5.904227399878436e-05, 0.0019265630519994376),
        3: (2.4723571999857087e-05, 0.007527870420001819, 5.854284899942286e-05, 0.0019207976360021348),
        4: (4.966531400077657e-05, 0.011915237750001325, 0.00011496014799740805, 0.004256961907999539),
        5: (4.922295499909525e-05, 0.012179142591001436, 0.00011955419199875906, 0.004155555682998057),
        6: (6.274842900165823e-05, 0.015590079618997155, 0.00013133193500107157, 0.0054354537500003065),
        7: (5.83403230011754e-05, 0.015320081203000883, 0.0001414624829967579, 0.005579915436000192),
        8: (6.105794600443914e-05, 0.0187047158159985, 0.00013606474100561172, 0.00796887394800251),
        9: (6.171482500212732e-05, 0.01802234846999636, 0.0001404538310016506, 0.007473570082998776),
        10: (6.912233100410959e-05, 0.022022588018997338, 0.000161577128996214, 0.009942683109999052),
        11: (0.00013744824700188474, 0.054509256608002044, 0.0002805470279981819, 0.026773343823999308),
        12: (0.00014333199600150692, 0.05313820058000238, 0.0002844032700013486, 0.026430248529999513),
        13: (0.0001869723169984354, 0.07050848679400223, 0.00044944160400700627, 0.03733307768499798),
        14: (0.0001888368189956964, 0.07119258922900008, 0.0004449215309959982, 0.037365963279002254),
        15: (0.00020653674299865083, 0.07925425235099103, 0.0004733462740041432, 0.04531428782700641),
        16: (0.00024536155700116073, 0.09283065392999926, 0.0005021318369963411, 0.05355130114100028),
        17: (0.00024055251099707676, 0.09472176539099927, 0.0005007462050034518, 0.054181309855001025),
        18: (0.0002750782849980169, 0.10861212989799969, 0.0005184747150033218, 0.06320860042899584),
        19: (0.0002739765879923653, 0.10776655998000204, 0.0005343634500040936, 0.06304528438299893),
        20: (0.00031259720299931356, 0.1201846706570086, 0.0005585428389986193, 0.07310007699800372),
        21: (0.00031195458100592076, 0.12023122594999772, 0.0005478160969942111, 0.07273125759499818),
        22: (0.00034954723399459916, 0.131901995987997, 0.0005851533209970511, 0.08201949736200367)
    }
    PLAINTEXT_MODULUS = 2**16 + 1

    def __init__(self, build_dir: str = "build") -> None:
        # Serialization
        self.compiled_dir: str = build_dir
        self.computation_dir: str = os.path.join(build_dir, "computation")
        self.display_dir: str = os.path.join(build_dir, "display")
        self.log_dir: str = os.path.join(build_dir, "logs")
        self.keys_dir: str = "keys"
        self.input_dir: str = "inputs"
        self.results_dir: str = "results"

        directories = [
            os.path.join(self.computation_dir, self.keys_dir),
            os.path.join(self.computation_dir, self.input_dir),
            os.path.join(self.computation_dir, self.results_dir),
            os.path.join(self.display_dir, self.keys_dir),
            self.log_dir,
        ]
        for dir in directories:
            os.makedirs(dir, exist_ok=True)

        self.serial_type: SERTYPE = BINARY
        self.serial_name: Name = Name("BINARY")

        # CryptoContext
        self.cc: CryptoContext | None = None
        self.cc_name: Name = Name("cc")
        self.cc_file: str = os.path.join(self.keys_dir, "crypto_context.enc")

        # Keys
        self.key_pair: KeyPair | None = None
        self.private_key_name: Name = Name("sk")
        self.private_key_file: str = os.path.join(
            self.keys_dir, "key_private.enc")
        self.mult_key_file: str = os.path.join(self.keys_dir, "key_mult.enc")
        self.rotation_key_file: str = os.path.join(
            self.keys_dir, "key_rotation.enc")

        # Keeping track of generated names
        self.name_counters: dict[str, int] = {}
        self.existing_names: set[str] = {}

    def compile(self, tree: Module, strategy: PackingStrategy,
                remove_preprocessing: bool = True) -> None:
        """Execute all compiler passes"""

        self.existing_names = self.collect_var_names(tree)
        self.save_program_to_file(
            tree, os.path.join(self.log_dir, "0_source.py"))

        mnf_ast = self.lower_to_mnf(tree)
        self.save_program_to_file(
            mnf_ast, os.path.join(self.log_dir, "1_mnf.py"))

        packing_ast = self.pack_data(mnf_ast, strategy)
        self.save_program_to_file(
            packing_ast, os.path.join(self.log_dir, "2_packing.py"))

        circuit_ast = self.lower_to_circuit(packing_ast)
        self.save_program_to_file(
            circuit_ast, os.path.join(self.log_dir, "3_circuit.py"))

        if remove_preprocessing:
            circuit_ast = self.remove_preprocessing(circuit_ast)
            self.save_program_to_file(
                circuit_ast, os.path.join(self.log_dir, "4_preprocessing.py")
            )

        self.init_crypto_context(circuit_ast)

        computation_ast, output_ast = self.encrypt_and_serialize(circuit_ast)
        self.save_program_to_file(
            computation_ast, os.path.join(self.log_dir, "5_computation.py"))
        self.save_program_to_file(
            computation_ast,
            os.path.join(self.computation_dir, "computation.py"))
        self.save_program_to_file(
            output_ast, os.path.join(self.display_dir, "display.py"))
        self.save_program_to_file(output_ast, os.path.join(
            self.log_dir, "5_display.py"))

    # --------------------- Lower to Monadic Normal Form ----------------------

    def lower_to_mnf(self, tree: Module) -> Module:
        """Convert the AST to monadic normal form."""
        match tree:
            case Module([imports, *body]):
                new_body = [stmts for st in body
                            for stmts in self.lower_to_mnf_stmt(st)]
                return Module([imports, *new_body], type_ignores=[])
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def lower_to_mnf_stmt(self, st: stmt) -> list[stmt]:
        match st:
            case Expr(Call(Name("print"), [exp])):
                new_exp, tmp = self.lower_to_mnf_exp(exp, True)
                new_stmts = [Assign([name], init_exp)
                             for name, init_exp in tmp]
                return new_stmts + [
                    Expr(Call(Name("print"), [new_exp], keywords=[]))
                ]
            case Expr(exp):
                return []
            case Assign([Name(var)], exp):
                new_exp, tmp = self.lower_to_mnf_exp(exp, False)
                new_stmts = [Assign([name], init_exp)
                             for name, init_exp in tmp]
                return new_stmts + [Assign([Name(var)], new_exp)]
            case _:
                raise SyntaxError(f"Invalid statement: {repr(st)}")

    def lower_to_mnf_exp(self, exp: expr,
                         need_atomic: bool) -> tuple[expr, Temporaries]:
        match exp:
            case Name(_):
                return exp, []
            case List(elts):
                new_elts = [self.remove_sub(e, False) for e in elts]
                new_exp, tmp = List(new_elts), []
            case Call(Name("Permutation"), [List(elts)]):
                new_exp, tmp = exp, []
            case BinOp(exp1, Mult(), exp2):
                atm1, tmp1 = self.lower_to_mnf_exp(exp1, True)
                atm2, tmp2 = self.lower_to_mnf_exp(exp2, True)
                new_exp = BinOp(atm1, Mult(), atm2)
                tmp = tmp1 + tmp2
            case Call(exp1, [exp2]):
                atm1, tmp1 = self.lower_to_mnf_exp(exp1, True)
                atm2, tmp2 = self.lower_to_mnf_exp(exp2, True)
                new_exp = Call(atm1, [atm2], keywords=[])
                tmp = tmp1 + tmp2
            case _:
                raise SyntaxError(f"Invalid expression: {repr(exp)}")

        if need_atomic:
            tmp_var = Name(self.generate_name())
            tmp.append((tmp_var, new_exp))
            new_exp = tmp_var

        return new_exp, tmp

    def remove_sub(self, exp: expr, signed: bool) -> expr:
        """Move subtractions inside constants."""
        match exp:
            case Constant(e):
                if isinstance(e, int):
                    return Constant(-e) if signed else Constant(e)
                else:
                    raise ValueError(f"Non-integer value: {e}")
            case UnaryOp(USub(), e):
                return self.remove_sub(e, not signed)

    # ------------------------------- Pack Data -------------------------------

    def pack_data(self, tree: Module, strategy: PackingStrategy) -> Module:
        """Pack data according to the given strategy."""
        match tree:
            case Module([imports, *body]):
                # Create new body and dataflow graph
                graph = nx.DiGraph()
                nodes: dict[str, Node] = {}
                node_count = count()
                new_body = [
                    self.pack_data_stmt(st, graph, nodes, node_count)
                    for st in body
                ]
                self.save_graph_to_file(
                    graph, os.path.join(self.log_dir, "graph.svg"))

                # Choose packings in graph
                self.choose_packing(graph, strategy)
                self.save_graph_to_file(
                    graph, os.path.join(self.log_dir, "packed_graph.svg"))

                # Acutally pack data
                roots = [
                    node for node in graph.nodes if graph.in_degree(node) == 0]
                for root in roots:
                    self.apply_packing(root.value)
                return Module([imports, *new_body], type_ignores=[])
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def pack_data_stmt(self, st: stmt, graph: nx.DiGraph,
                       nodes: dict[str, Node], node_count: count) -> stmt:
        """Create a `Packing` for constants and add the statement to the graph.

        Constant values are replaced by tuples that consist of a `Packing`
        object and the value. The tuple becomes a root node in the graph and can
        be referenced from there. Operations become inner nodes of the graph.
        """
        match st:
            case Assign([Name(var)], exp):
                exp, node = self.pack_data_exp(exp, nodes, node_count, graph)
                nodes[var] = node
                return Assign([Name(var)], exp)
            case Expr(_):
                return st
            case _:
                raise SyntaxError(f"Invalid statement: {repr(st)}")

    def pack_data_exp(self, exp: expr, nodes: dict[str, Node],
                      node_count: count, graph: nx.DiGraph) -> tuple[expr, Node]:
        match exp:
            case Name(var):
                node = nodes[var]
                new_exp = exp
            case List(elts):
                new_exp = Tuple([
                    Constant(Packing(DataType.LIST, None, len(elts), None)),
                    List(elts)
                ])
                node = Node(next(node_count), new_exp)
                graph.add_node(node)
            case Call(Name("Permutation"), [List(elts)]):
                new_exp = Tuple([
                    Constant(
                        Packing(DataType.PERMUTATION, None, len(elts), None)
                    ),
                    List(elts)
                ])
                node = Node(next(node_count), new_exp)
                graph.add_node(node)
            case Call(Name("inv"), [Name(permutation)]):
                new_exp = exp
                node = Node(next(node_count), Operation.INV)
                graph.add_node(node)
                graph.add_edge(nodes[permutation], node)
            case Call(Name(permutation), [Name(array)]):
                new_exp = exp
                node = Node(next(node_count), Operation.APPLY)
                graph.add_node(node)
                graph.add_edge(nodes[permutation], node)
                graph.add_edge(nodes[array], node)
            case BinOp(Name(permutation1), Mult(), Name(permutation2)):
                new_exp = exp
                node = Node(next(node_count), Operation.COMPOSE)
                graph.add_node(node)
                graph.add_edge(nodes[permutation1], node)
                graph.add_edge(nodes[permutation2], node)
            case _:
                raise SyntaxError(f"Invalid expression: {repr(exp)}")
        return new_exp, node

    def choose_packing(self, graph: nx.DiGraph,
                       strategy: PackingStrategy) -> None:
        match strategy:
            case PackingStrategy.ROW:
                roots = [
                    node for node in graph.nodes if graph.in_degree(node) == 0]
                for root in roots:
                    root.value.elts[0].value.layout = Layout.ROW
            case PackingStrategy.DIAG:
                roots = [
                    node for node in graph.nodes if graph.in_degree(node) == 0]
                for root in roots:
                    root.value.elts[0].value.layout = Layout.DIAG
            case PackingStrategy.CUSTOM:
                self.choose_packing(graph, PackingStrategy.ROW)
                row_cost = self.approximate_cost(graph)
                logging.info(f"Approximated ROW cost: {row_cost:.2f}")

                self.choose_packing(graph, PackingStrategy.DIAG)
                diag_cost = self.approximate_cost(graph)
                logging.info(f"Approximated DIAG cost: {diag_cost:.2f}")

                if row_cost <= diag_cost:
                    self.choose_packing(graph, PackingStrategy.ROW)
                    logging.info("Chosen packing: ROW")
                else:
                    logging.info("Chosen packing: DIAG")
            case _:
                raise TypeError(f"Unkown packing strategy: {repr(strategy)}")

    def approximate_cost(self, graph: nx.DiGraph) -> float:
        """Estimate the computational work for a packing assignment."""
        packings: dict[Node, Packing] = {}
        depths: dict[Node, int] = {}
        op_counts = {"add": 0, "mult": 0, "cmult": 0, "rot": 0}
        for node in nx.topological_sort(graph):
            preds = list(graph.predecessors(node))
            if len(preds) == 0:  # Root node
                packing = node.value.elts[0].value
                if packing.layout == Layout.ROW:  # Compute padded size
                    padded_size = 2**math.ceil(math.log2(packing.size))
                else:
                    padded_size = packing.size
                packings[node] = Packing(packing.data_type, packing.layout,
                                         packing.size, padded_size)
                depths[node] = 0
            elif len(preds) <= 2:  # Operation node
                packing = packings[preds[0]]
                new_depth, new_op_counts = self.get_operation_metrics(
                    node.value, packing)
                if len(preds) == 1:
                    depth = new_depth + depths[preds[0]]
                else:
                    depth = new_depth + max(depths[preds[0]], depths[preds[1]])
                for op in new_op_counts:
                    op_counts[op] += new_op_counts[op]
                packings[node] = packing
                depths[node] = depth
            else:
                raise ValueError(
                    f"Unexpected amount of preds for {node}: {len(preds)}")
        depth = max(depths.values(), default=0)
        if depth in self.COST_BY_DEPTH:
            add_cost, mult_cost, cmult_cost, rot_cost = self.COST_BY_DEPTH[depth]
        else:
            # Depth probably not supported
            add_cost, mult_cost, cmult_cost, rot_cost = 1000, 1000, 1000, 1000
        cost = (op_counts["add"] * add_cost + op_counts["mult"] * mult_cost
                + op_counts["cmult"] * cmult_cost + op_counts["rot"] * rot_cost)
        return cost

    def get_operation_metrics(self, operation: Operation,
                              packing: Packing) -> tuple[int, dict[str, int]]:
        """Compute the depth, adds, mults, cmults and rots of a given operation."""
        layout = packing.layout
        padded_size = packing.padded_size
        if layout == Layout.ROW:
            if operation == Operation.APPLY or operation == Operation.COMPOSE:
                adds = 2 * padded_size * \
                    math.log2(padded_size) + padded_size - 1
                mults = padded_size
                cmults = 2 * padded_size
                rots = 2 * padded_size * (1 + math.log2(padded_size)) - 2
                depth = 2
            elif operation == Operation.INV:
                size = packing.size
                adds = size * 2 * math.log2(padded_size) + (size - 1)
                mults = 1
                cmults = 2 * (size - 1)
                rots = size * 2 * math.log2(padded_size)
                depth = 3
            else:
                raise ValueError(f"Unknown operation: {operation}")
        elif layout == Layout.DIAG:
            cmults = 0
            if operation == Operation.APPLY:
                adds = padded_size
                mults = padded_size
                rots = padded_size
                depth = 1
            elif operation == Operation.COMPOSE:
                adds = padded_size**2
                mults = padded_size**2
                rots = padded_size**2
                depth = 1
            elif operation == Operation.INV:
                size = packing.size
                padded_size = 2**math.ceil(math.log2(size))
                adds = size * 2 * math.log2(padded_size) + (size - 1) + size**2
                mults = 1
                cmults = 2 * (size - 1) + size ** 2
                rots = size * 2 * math.log2(padded_size) + (size**2 - 1)
                depth = 3
            else:
                raise ValueError(f"Unknown operation: {operation}")
        else:
            raise ValueError(f"Unknown layout: {layout}")
        return depth, {"add": adds, "mult": mults, "cmult": cmults, "rot": rots}

    def apply_packing(self, exp: Tuple) -> None:
        match exp:
            case Tuple([Constant(Packing(DataType.LIST, Layout.ROW, _, _)), List(elts)]):
                padded_array = pad_array_to_matrix([e.value for e in elts])
                packed_array, padded_size = pack_matrix_row(padded_array)
                new_elts = [Constant(i) for i in packed_array]
            case Tuple([Constant(Packing(DataType.PERMUTATION, Layout.ROW, _, _)), List(elts)]):
                matrix = to_matrix([e.value for e in elts])
                packed_matrix, padded_size = pack_matrix_row(matrix)
                new_elts = [Constant(i) for i in packed_matrix]
            case Tuple([Constant(Packing(DataType.LIST, Layout.DIAG, _, _)), List(elts)]):
                new_elts, padded_size = elts, len(elts)
            case Tuple([Constant(Packing(DataType.PERMUTATION, Layout.DIAG, _, _)), List(elts)]):
                padded_size = len(elts)
                matrix = to_matrix([e.value for e in elts])
                packed_matrix = pack_matrix_diag(matrix)
                new_elts = [List([Constant(i) for i in diag])
                            for diag in packed_matrix]
            case _:
                raise TypeError(f"Expression has incorrect type: {repr(exp)}")
        exp.elts[0].value.padded_size = padded_size
        exp.elts[1].elts = new_elts

    # --------------------------- Lower to Circuit ----------------------------

    def lower_to_circuit(self, tree: Module) -> Module:
        """Express the AST as an arithmetic circuit."""
        match tree:
            case Module([imports, *body]):
                packings: dict[str, Packing] = {}
                new_body = [
                    stmts for st in body
                    for stmts in self.lower_to_circuit_stmt(st, packings)
                ]
                return Module([imports, *new_body], type_ignores=[])
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def lower_to_circuit_stmt(self, st: stmt,
                              packings: dict[str, Packing]) -> list[stmt]:
        """Map an operation to its homomorphic counterpart(s)."""
        match st:
            case Expr(exp):
                return [st]
            case Assign([Name(var)], exp):
                new_exp, tmp, packings[var] = self.lower_to_circuit_exp(
                    exp, packings)
                new_stmts = [Assign([name], init_exp)
                             for name, init_exp in tmp]
                return new_stmts + [Assign([Name(var)], new_exp)]
            case _:
                raise SyntaxError(f"Invalid statement: {repr(st)}")

    def lower_to_circuit_exp(
        self,
        exp: expr,
        packings: dict[str, Packing]
    ) -> tuple[expr, Temporaries, Packing]:
        """Determine the homomorphic counterpart of an expression.

        Function calls are mapped to homomorphic routines based on packings.
        Possible preprocessing assignments are returned as Temporaries.
        """
        tmp = []
        match exp:
            case Name(var):
                return exp, tmp, packings[var]
            case Tuple([Constant(packing), List(_)]):
                return exp, tmp, packing
            case BinOp(Name(permutation1), Mult(), Name(permutation2)):
                packing = packings[permutation1]
                new_packing = packing
                match packing.layout:
                    case Layout.ROW:
                        A_hat_name = Name(self.generate_name())
                        A_hat_exp = Call(
                            Name("preprocess_left_matrix_row"),
                            [self.cc_name, Name(permutation1),
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                        tmp.append((A_hat_name, A_hat_exp))
                        B_hat_name = Name(self.generate_name())
                        B_hat_exp = Call(
                            Name("preprocess_right_matrix_row"),
                            [self.cc_name, Name(permutation2),
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                        tmp.append((B_hat_name, B_hat_exp))
                        new_exp = Call(
                            Name("matrix_matrix_mult_row"),
                            [self.cc_name, A_hat_name, B_hat_name,
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                    case Layout.DIAG:
                        B_hat_name = Name(self.generate_name())
                        B_hat_exp = Call(
                            Name("preprocess_matrix_diag"),
                            [self.cc_name, Name(permutation2)],
                            keywords=[]
                        )
                        tmp.append((B_hat_name, B_hat_exp))
                        new_exp = Call(
                            Name("matrix_matrix_mult_diag"),
                            [self.cc_name, Name(permutation1), B_hat_name],
                            keywords=[]
                        )
                    case _:
                        raise ValueError(
                            f"Multiplication not supported for "
                            f"layout {packing.layout}")
            case Call(Name("inv"), [Name(permutation)]):
                packing = packings[permutation]
                new_packing = Packing(DataType.INT, None, None, None)
                if packing.layout == Layout.DIAG:
                    # Next power of 2
                    padded_size = 2**math.ceil(math.log2(packing.size))
                    P_hat_name = Name(self.generate_name())
                    P_hat_exp = Call(
                        Name("convert_diag_to_row"),
                        [self.cc_name, Name(permutation)],
                        keywords=[]
                    )
                    tmp.append((P_hat_name, P_hat_exp))
                    permutation = P_hat_name.id
                else:
                    padded_size = packing.padded_size
                new_exp = Call(
                    Name("inversion_number_row"),
                    [self.cc_name, Name(permutation),
                     Constant(packing.size), Constant(padded_size)],
                    keywords=[]
                )
            case Call(Name(permutation), [Name(array)]):
                packing = packings[permutation]
                new_packing = packings[array]
                match packing.layout:
                    case Layout.ROW:
                        A_hat_name = Name(self.generate_name())
                        A_hat_exp = Call(
                            Name("preprocess_left_matrix_row"),
                            [self.cc_name, Name(permutation),
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                        tmp.append((A_hat_name, A_hat_exp))
                        v_hat_name = Name(self.generate_name())
                        v_hat_exp = Call(
                            Name("preprocess_right_matrix_row"),
                            [self.cc_name, Name(array),
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                        tmp.append((v_hat_name, v_hat_exp))
                        new_exp = Call(
                            Name("matrix_matrix_mult_row"),
                            [self.cc_name, A_hat_name, v_hat_name,
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                    case Layout.DIAG:
                        v_hat_name = Name(self.generate_name())
                        v_hat_exp = Call(
                            Name("preprocess_vector_diag"),
                            [self.cc_name, Name(array),
                             Constant(packing.padded_size)],
                            keywords=[]
                        )
                        tmp.append((v_hat_name, v_hat_exp))
                        new_exp = Call(
                            Name("matrix_vector_mult_diag"),
                            [self.cc_name, Name(permutation), v_hat_name],
                            keywords=[]
                        )
                    case _:
                        raise ValueError(
                            f"Multiplication not supported for "
                            f"layout {packing.layout}")
            case _:
                raise SyntaxError(f"Invalid expression: {repr(exp)}")

        return new_exp, tmp, new_packing

    # ------------------------- Remove Preprocessing --------------------------

    def remove_preprocessing(self, tree: Module) -> None:
        """Remove duplicate preprocessing calls."""
        match tree:
            case Module([imports, *body]):
                # Preprocessed versions of a variable
                preprocessed: dict[str, dict[str, str]] = {}
                new_body = [self.remove_preprocessing_stmt(st, preprocessed)
                            for st in body]
                return Module([imports, *new_body], type_ignores=[])
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def remove_preprocessing_stmt(
        self,
        st: stmt,
        preprocessed: dict[str, dict[str, str]]
    ) -> stmt:
        """Remove duplicate preprocessing call from the statement.

        If a variable is newly assigned, its value in the dict will be removed.
        """
        match st:
            case Expr(Call(Name("print"), [Name(var)])):
                return st
            case Assign([Name(var)], exp):
                # Remove from dict if newly assigned
                preprocessed.pop(var, None)

                # Assign possibly new expression
                new_exp = self.remove_preprocessing_exp(var, exp, preprocessed)
                return Assign([Name(var)], new_exp)
            case _:
                raise SyntaxError(f"Invalid statement: {repr(st)}")

    def remove_preprocessing_exp(
        self,
        var: str,
        exp: expr,
        preprocessed: dict[str, dict[str, str]]
    ) -> expr:
        """Return an expression containing the result of exp.

        If exp is a previously computed preprocessing call for a variable that
        has not changed since, this will return the variable which contains the
        previous result. Otherwise, exp will be returned. The dictionary is also
        updated if a new preprocessing call is found or a variable is assigned.
        """
        match exp:
            case (Tuple([Constant(_), List(_)])
                  | Call(Name("matrix_matrix_mult_row"))
                  | Call(Name("matrix_vector_mult_diag"))
                  | Call(Name("matrix_matrix_mult_diag"))
                  | Call(Name("inversion_number_row"))
                  | Call(Name("convert_diag_to_row"))):
                return exp
            case Name(var2):
                if value := preprocessed.get(var2):
                    preprocessed[var] = value
                return exp
            case Call(Name("preprocess_left_matrix_row"),
                      [_, Name(matrix), _]):
                operation = "preprocess_left_matrix_row"
                prep_var = matrix
            case Call(Name("preprocess_right_matrix_row"),
                      [_, Name(matrix), _]):
                operation = "preprocess_right_matrix_row"
                prep_var = matrix
            case Call(Name("preprocess_vector_diag"), [_, Name(vector), _]):
                operation = "preprocess_vector_diag"
                prep_var = vector
            case Call(Name("preprocess_matrix_diag"), [_, Name(matrix)]):
                operation = "preprocess_matrix_diag"
                prep_var = matrix
            case _:
                raise SyntaxError(f"Invalid expression: {repr(exp)}")

        if previous := preprocessed.get(prep_var, {}).get(operation):
            return Name(previous)
        preprocessed.setdefault(prep_var, {})[operation] = var
        return exp

    # ----------------------- Initialize Crypto Context -----------------------

    def init_crypto_context(self, tree: Module) -> None:
        """Create and serialize the `CryptoContext` together with its keys."""
        mult_depth, rotations = self.determine_parameters(tree)
        logging.info(f"Determined multiplicative depth: {mult_depth}")
        logging.info(f"Determined rotation indices: {rotations}")

        if mult_depth > 22:
            raise ValueError(f"Depth {mult_depth} is too big.")

        parameters = CCParamsBFVRNS()
        parameters.SetPlaintextModulus(self.PLAINTEXT_MODULUS)
        parameters.SetMultiplicativeDepth(mult_depth)

        self.cc = GenCryptoContext(parameters)
        logging.info(
            f"Chosen plaintext modulus p: {self.cc.GetPlaintextModulus()}")
        logging.info(f"Chosen ring dimension N: {self.cc.GetRingDimension()}")
        logging.info(
            f"Chosen cyclotomic number 2N: {self.cc.GetCyclotomicOrder()}")
        logging.info(f"Chosen ciphertext modulus q: {self.cc.GetModulus()}")
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        self.key_pair = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.key_pair.secretKey)
        self.cc.EvalRotateKeyGen(self.key_pair.secretKey, list(rotations))

        # Store computation keys to files
        cc_path = os.path.join(self.computation_dir, self.cc_file)
        if not SerializeToFile(cc_path, self.cc, self.serial_type):
            raise IOError(f"Error serializing CryptoContext to {cc_path}")

        mult_path = os.path.join(self.computation_dir, self.mult_key_file)
        if not self.cc.SerializeEvalMultKey(mult_path, self.serial_type):
            raise IOError(f"Error serializing mult key to {mult_path}")

        rotation_path = os.path.join(
            self.computation_dir, self.rotation_key_file)
        if not self.cc.SerializeEvalAutomorphismKey(
                rotation_path, self.serial_type):
            raise IOError(f"Error serializing rotation key to {rotation_path}")

        # Store display keys to files
        cc_path = os.path.join(self.display_dir, self.cc_file)
        if not SerializeToFile(cc_path, self.cc, self.serial_type):
            raise IOError(f"Error serializing CryptoContext to {cc_path}")

        sk_path = os.path.join(self.display_dir, self.private_key_file)
        if not SerializeToFile(
                sk_path, self.key_pair.secretKey, self.serial_type):
            raise IOError(f"Error serializing private key to {sk_path}")

        # Clear cached crypto data
        ClearEvalMultKeys()
        self.cc.ClearEvalAutomorphismKeys()
        ReleaseAllContexts()

    def determine_parameters(self, tree: Module) -> tuple[int, set[int]]:
        """Determine the required multiplicative depth and rotation indices."""
        match tree:
            case Module([imports, *body]):
                # Padded sizes and multiplicative levels
                vars: dict[str, tuple[int, int]] = {}
                # Required rotation indices
                rotations: set[int] = set()
                for st in body:
                    self.determine_parameters_stmt(st, vars, rotations)
                mult_depth = max((tup[1] for tup in vars.values()), default=0)
                return mult_depth, rotations
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def determine_parameters_stmt(self, st: stmt,
                                  vars: dict[str, tuple[int, int]],
                                  rotations: set[int]) -> None:
        """Update `vars` and `rotations` based on the statement."""
        match st:
            case Expr(Call(Name("print"), [Name(var)])):
                return
            case Assign([Name(var)], exp):
                vars[var] = self.determine_parameters_exp(
                    exp, vars, rotations)
            case _:
                raise SyntaxError(f"Invalid statement: {repr(st)}")

    def determine_parameters_exp(self, exp: expr,
                                 vars: dict[str, tuple[int, int]],
                                 rotations: set[int]) -> int:
        """Add the rotation indices required by an expression to the given set.

        Returns:
            The padded size and multiplicative level of the expression's result.
        """
        match exp:
            case Tuple([Constant(packing), List(_)]):
                return packing.padded_size, 0
            case Name(var):
                return vars[var]
            case Call(Name("preprocess_left_matrix_row"),
                      [_, Name(matrix), _]):
                size, level = vars[matrix]
                rotations.update(
                    range(1, size),
                    (-2**i for i in range(int(math.log2(size))))
                )
                return size, level + 1
            case Call(Name("preprocess_right_matrix_row"),
                      [_, Name(matrix), _]):
                size, level = vars[matrix]
                rotations.update(
                    (i * size for i in range(1, size)),
                    (-2**i * size for i in range(int(math.log2(size))))
                )
                return size, level + 1
            case Call(Name("matrix_matrix_mult_row"),
                      [_, Name(matrix1), Name(matrix2), _]):
                size, matrix1_level = vars[matrix1]
                _, matrix2_level = vars[matrix2]
                return size, max(matrix1_level, matrix2_level) + 1
            case Call(Name("preprocess_vector_diag"), [_, Name(vector), _]):
                size, level = vars[vector]
                rotations.add(-size)
                return size, level
            case Call(Name("matrix_vector_mult_diag"),
                      [_, Name(matrix), Name(vector)]):
                size, matrix_level = vars[matrix]
                _, vector_level = vars[vector]
                rotations.update(range(1, size))
                return size, max(matrix_level, vector_level) + 1
            case Call(Name("preprocess_matrix_diag"), [_, Name(matrix)]):
                size, level = vars[matrix]
                rotations.add(-size)
                return size, level
            case Call(Name("matrix_matrix_mult_diag"),
                      [_, Name(matrix1), Name(matrix2)]):
                size, matrix1_level = vars[matrix1]
                _, matrix2_level = vars[matrix2]
                rotations.update(range(1, size))
                return size, max(matrix1_level, matrix2_level) + 1
            case Call(Name("convert_diag_to_row"), [_, Name(matrix)]):
                size, matrix_level = vars[matrix]
                padded_size = 2**math.ceil(math.log2(size))
                rotations.update(
                    i * (1 - padded_size) - j for j in range(size)
                    for i in range(size) if not (i == j == 0)
                )
                return padded_size, matrix_level + 1
            case Call(Name("inversion_number_row"), [_, Name(matrix), _, _]):
                size, matrix_level = vars[matrix]
                rotations.update(
                    2**i for i in range(int(math.log2(size**2)))
                )
                return 1, matrix_level + 3
            case _:
                raise SyntaxError(f"Invalid expression: {repr(exp)}")

    # ------------------------- Encrypt and Serialize -------------------------

    def encrypt_and_serialize(self, tree: Module) -> tuple[Module, Module]:
        """Encrypt all constant values and store them in files.

        Returns:
            A tuple consisting of two programs.
            - The first program can be sent to an untrusted party for computation.
            - The second program can dsiplay the results.
        """
        match tree:
            case Module([imports, *body]):
                # Store variable packings
                packings: dict[str, Packing] = {}

                # Split the program into two types of statements
                computation: list[stmt] = []
                display: list[stmt] = []
                for st in body:
                    c_st, d_st = self.encrypt_and_serialize_stmt(st, packings)
                    computation.append(c_st)
                    if d_st:
                        display.append(d_st)

                # Add prelude
                computation = [
                    Assign([self.cc_name], Call(
                        Name("deserialize_crypto_context"),
                        [Constant(self.cc_file), self.serial_name],
                        keywords=[]
                    )),
                    Expr(Call(
                        Name("deserialize_mult_key"),
                        [Constant(self.mult_key_file), self.serial_name,
                         self.cc_name],
                        keywords=[]
                    )),
                    Expr(Call(
                        Name("deserialize_rotation_key"),
                        [Constant(self.rotation_key_file), self.serial_name,
                         self.cc_name],
                        keywords=[]
                    )),
                    *computation
                ]
                display = [
                    Assign([self.cc_name], Call(
                        Name("deserialize_crypto_context"),
                        [Constant(self.cc_file), self.serial_name],
                        keywords=[]
                    )),
                    Assign([self.private_key_name], Call(
                        Name("deserialize_private_key"),
                        [Constant(self.private_key_file), self.serial_name],
                        keywords=[]
                    )),
                    *display
                ]

                return (Module([imports, *computation], type_ignores=[]),
                        Module([imports, *display], type_ignores=[]))
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def encrypt_and_serialize_stmt(
        self,
        st: stmt,
        packings: dict[str, Packing]
    ) -> tuple[stmt, stmt | None]:
        """Split a statement into computation and display.

        The statement for computation
        - loads encrypted data
        - or computes an equivalent function in the encrypted domain
        - or stores ciphertext in a file.

        The statement for display prints a stored ciphertext (if required).

        Returns:
            A tuple consisting of the two statements for computation and display.
        """
        match st:
            case Expr(Call(Name("print"), [Name(var)])):
                return self.serialize_print_stmt(var, packings[var])
            case Assign([Name(var)], exp):
                new_exp, packings[var] = self.encrypt_and_serialize_exp(
                    exp, packings)
                return Assign([Name(var)], new_exp), None
            case _:
                raise SyntaxError(f"Invalid statement: {repr(st)}")

    def serialize_print_stmt(self, var: str,
                             packing: Packing) -> tuple[stmt, stmt]:
        """Generate statements for serializing / printing a ciphertext."""
        match packing:
            case (Packing(layout=Layout.ROW)
                  | Packing(data_type=DataType.INT)
                  | Packing(data_type=DataType.LIST, layout=Layout.DIAG)):
                file_name = Constant(os.path.join(
                    self.results_dir,
                    self.generate_name("output", False) + ".enc"
                ))
                store_call = Name("serialize")
                store_args = [file_name, Name(var), self.serial_name]
                print_call = Name("print_ciphertext")
                print_args = [file_name, Constant(packing),
                              self.serial_name, self.cc_name,
                              self.private_key_name]
            case Packing(data_type=DataType.PERMUTATION, layout=Layout.DIAG):
                file_names = [Constant(os.path.join(
                    self.results_dir,
                    self.generate_name("output", False) + ".enc"
                )) for _ in range(packing.padded_size)]
                store_call = Name("serialize_many")
                store_args = [List(file_names), Name(var), self.serial_name]
                print_call = Name("print_many")
                print_args = [List(file_names), Constant(packing),
                              self.serial_name, self.cc_name,
                              self.private_key_name]
            case _:
                raise ValueError(f"Unrecognized layout {packing.layout}")
        store_st = Expr(Call(store_call, store_args, keywords=[]))
        print_st = Expr(Call(print_call, print_args, keywords=[]))
        return store_st, print_st

    def encrypt_and_serialize_exp(
            self,
            exp: expr,
            packings: dict[str, Packing]
    ) -> tuple[expr, Packing]:
        """Encrypt an expression if it contains constant data.

        In that case, the data is encrypted and written to a file.

        Returns:
            A tuple consisting of
            - an expression that loads the data
            - the packing of the data stored in the expression.
        """
        match exp:
            case Tuple([Constant(packing), List(elts)]):
                match packing.layout:
                    case Layout.ROW:
                        match packing.data_type:
                            case DataType.LIST:
                                file_name = self.encrypt_to_file(elts)
                            case DataType.PERMUTATION:
                                file_name = self.encrypt_to_file(elts)
                        call_name = "deserialize"
                    case Layout.DIAG:
                        match packing.data_type:
                            case DataType.LIST:
                                file_name = self.encrypt_to_file(elts)
                                call_name = "deserialize"
                            case DataType.PERMUTATION:
                                file_name = List([self.encrypt_to_file(l.elts)
                                                  for l in elts])
                                call_name = "deserialize_many"
                    case _:
                        raise ValueError(
                            f"Unrecognized layout {packing.layout}")
                new_exp = Call(
                    Name(call_name),
                    [file_name, self.serial_name],
                    keywords=[]
                )
                return new_exp, packing
            case Name(var):
                return exp, packings[var]
            case Call(Name("preprocess_left_matrix_row"), [_, Name(matrix), _]):
                return exp, packings[matrix]
            case Call(Name("preprocess_right_matrix_row"), [_, Name(matrix), _]):
                return exp, packings[matrix]
            case Call(Name("matrix_matrix_mult_row"), [_, _, Name(matrix2), _]):
                return exp, packings[matrix2]
            case Call(Name("preprocess_vector_diag"), [_, Name(vector), _]):
                return exp, packings[vector]
            case Call(Name("matrix_vector_mult_diag"), [_, _, Name(vector)]):
                return exp, packings[vector]
            case Call(Name("preprocess_matrix_diag"), [_, Name(matrix)]):
                return exp, packings[matrix]
            case Call(Name("matrix_matrix_mult_diag"), [_, Name(matrix), _]):
                return exp, packings[matrix]
            case Call(Name("inversion_number_row"), [_, Name(matrix), _, _]):
                return exp, Packing(DataType.INT, None, None, None)
            case Call(Name("convert_diag_to_row"), [_, Name(matrix)]):
                packing = packings[matrix]
                padded_size = 2**math.ceil(math.log2(packing.size))
                return exp, Packing(packing.data_type, Layout.ROW,
                                    packing.size, padded_size)
            case _:
                raise SyntaxError(f"Invalid expression: {repr(exp)}")

    def encrypt_to_file(self, elts: list[Constant]) -> Constant:
        """Encrypt an array of integer constants and store it in a file.

        Returns:
            The string constant for the file name.
        """
        values = [e.value for e in elts]

        ciphertext = self.cc.Encrypt(
            self.key_pair.publicKey, self.cc.MakePackedPlaintext(values)
        )
        filename = os.path.join(
            self.input_dir, self.generate_name("input", False) + ".enc")
        file_path = os.path.join(self.computation_dir, filename)
        if not SerializeToFile(file_path, ciphertext, self.serial_type):
            raise IOError(
                f"Error writing serialization of {ciphertext} to {file_path}")
        return Constant(filename)

    # --------------------------------- Utils ---------------------------------

    def collect_var_names(self, tree: Module) -> set[str]:
        """Collect all variable names that occur in the source program.

        If a variable name conflicts with a name that is reserved for the target
        language, an error is raised.
        """
        match tree:
            case Module([imports, *body]):
                # Define existing names
                reserved_names = {
                    self.serial_name.id,
                    self.cc_name.id,
                    self.private_key_name.id,
                    *(name for name, _ in inspect.getmembers(
                        interpreter,
                        lambda x: inspect.isfunction(x) or inspect.isclass(x)
                    ))
                }
                # Collect names
                collected_names = set()
                for st in body:
                    match st:
                        case Assign([Name(var)]):
                            collected_names.add(var)
                # Check for duplicates
                shared_names = reserved_names & collected_names
                if shared_names:
                    raise NameError(
                        f"Source program uses reserved name: {shared_names}")
                return collected_names
            case _:
                raise TypeError(f"Excepted ast.Module, got {type(tree)}")

    def generate_name(self, prefix: str = "tmp",
                      check_duplicate: bool = True) -> str:
        """Generate a unique name of the form prefix{i} with an integer i.

        If the name will not be used for a variable (e.g. a file name),
        `check_duplicate` can be disabled.
        """
        counter = self.name_counters.get(prefix, 0)
        name = f"{prefix}{counter}"

        while check_duplicate and name in self.existing_names:
            counter += 1
            name = f"{prefix}{counter}"

        self.name_counters[prefix] = counter + 1
        return name

    def save_program_to_file(self, tree: Module, filename: str) -> None:
        code = unparse(fix_missing_locations(tree))
        with open(filename, "w") as f:
            f.write(code)

    def save_graph_to_file(self, graph: nx.DiGraph, filename: str) -> None:
        for node in graph.nodes:
            if node.value == Operation.APPLY:
                graph.nodes[node]["label"] = "apply"
            elif node.value == Operation.COMPOSE:
                graph.nodes[node]["label"] = "*"
            elif node.value == Operation.INV:
                graph.nodes[node]["label"] = "inv"
            else:
                packing = node.value.elts[0].value
                if packing.data_type == DataType.PERMUTATION:
                    label = "Permutation"
                else:
                    label = "Vector"
                label += f", size={packing.size}"
                if packing.layout == Layout.ROW:
                    label += ",\nlayout=row"
                elif packing.layout == Layout.DIAG:
                    label += ",\nlayout=diag"
                graph.nodes[node]["label"] = label
        pydot_graph = to_pydot(graph)
        pydot_graph.write(filename, format=os.path.splitext(filename)[1][1:])
