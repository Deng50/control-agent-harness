"""Static support checks for explicit numeric MATLAB snapshot functions.

This deliberately small language/interface subset is not a MATLAB parser or a
security sandbox.  No MATLAB is started and no user code is evaluated.  The
native adapter still verifies function resolution, repeated-input behaviour,
finite shapes and the actual terminal state of its dedicated MATLAB session.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat

from .contracts import ContractError


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?[ij]?")
_SHA = re.compile(r"[0-9a-fA-F]{64}")
_KEYWORDS = set("function end if elseif else for while switch case otherwise break continue return try catch arguments".split())
_PURE = set("""abs acos acosh acot acoth acsc acsch asec asech asin asinh atan atan2 atanh
ceil cos cosh cot coth csc csch exp expm expm1 fix floor hypot log log10 log1p log2 mod rem
round sec sech sign sin sinh sqrt tan tanh nthroot power real imag conj angle complex
plus minus times mtimes rdivide mrdivide ldivide mldivide mpower uplus uminus transpose ctranspose
zeros ones eye diag tril triu trace det rank norm cond rcond inv pinv chol ldl lu qr svd eig
linsolve sylvester lyap dlyap kron cross dot sum prod mean median min max cumsum cumprod diff gradient
cummax cummin sort sortrows unique union intersect setdiff setxor any all find nnz nonzeros
size length numel ndims isempty isscalar isvector isrow iscolumn ismatrix isnumeric isfloat
isreal isfinite isnan isinf islogical isequal isequaln isa class isstruct iscell isfield fieldnames
reshape squeeze permute ipermute repmat repelem cat horzcat vertcat flip fliplr flipud rot90
meshgrid ndgrid linspace logspace colon sub2ind ind2sub struct cell cell2mat num2cell num2str
double single logical int8 int16 int32 int64 uint8 uint16 uint32 uint64 typecast
bitand bitor bitxor bitcmp bitshift not and or xor assert error disp sprintf strcmp strcmpi
strncmp strncmpi char string strlength strtrim strrep contains startsWith endsWith lower upper
mustBeFinite mustBeReal mustBeNumeric mustBeInteger mustBePositive mustBeNonnegative
mustBeNonzero mustBeNonNan mustBeScalarOrEmpty mustBeVector mustBeA mustBeSize
true false pi eps inf Inf nan NaN realmin realmax intmin intmax i j""".split())
# Reject these identifiers outside strings/comments/field names even when used
# as a bare no-argument function or command, not only when followed by "(".
_FORBIDDEN = set("""persistent global eval evalin assignin clear clearvars run load save system unix dos
winopen web webread webwrite websave urlread urlwrite import javaaddpath javarmpath javaobject javamethod
py python pyrun pyrunfile actxserver actxgetrunningserver calllib loadlibrary unloadlibrary libpointer
mex mexext coder simulink set_param get_param sim parsim load_system open_system bdclose
setappdata getappdata guidata gcf gca figure timer tic toc datetime now clock rand randi randn rng
diary cd pwd addpath rmpath path restoredefaultpath rehash which exist fopen fclose fread fwrite
fprintf fscanf fileread readmatrix writematrix readtable writetable copyfile movefile delete mkdir rmdir
dir ls type edit keyboard dbstop pause input quit exit warning oncleanup handle parallel parfeval
batch spmd parfor parpool gpuarray tall distributed matfile h5read h5write feval str2func func2str
builtin functions classdef properties methods events enumeration metaclass""".split())
_OPAQUE = {".p", ".mlx", ".dll", ".so", ".dylib", ".jar", ".class"}
_EXPERT_NUMERIC = frozenset({"quadprog", "optimoptions", "speye", "spdiags"})


def _relative(value):
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive:
        raise ContractError("MATLAB source inventory requires confined relative file paths")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ContractError("MATLAB source inventory requires confined relative file paths")
    return path.as_posix()


def _checked(root, relative):
    path = root / relative
    for item in [root, *reversed([p for p in path.parents if p != root and p.is_relative_to(root)]), path]:
        try:
            info = item.lstat()
        except OSError as error:
            raise ContractError(f"MATLAB source snapshot dependency is missing: {relative}") from error
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ContractError("MATLAB source snapshots cannot follow filesystem aliases")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ContractError("MATLAB source dependency must be a regular file inside its snapshot")
    return path


def _tokens(text, label):
    """Lex only what support checks need, retaining locations and transpose.

    Single quotes adjacent to expression-ending tokens are transpose; other
    single quotes and double quotes are literals with doubled-quote escaping.
    Ambiguous whitespace transpose forms are outside this explicit subset.
    """
    tokens, index, line, column, block = [], 0, 1, 1, 0
    length = len(text)
    while index < length:
        char = text[index]
        if block:
            end = text.find("\n", index)
            end = length if end < 0 else end
            marker = text[index:end].strip()
            if marker == "%{": block += 1
            elif marker == "%}": block -= 1
            index = end
            if index < length:
                index += 1; line += 1; column = 1
            continue
        if char in " \t\r": index += 1; column += 1; continue
        if char == "\n":
            tokens.append(("newline", "\n", line, column)); index += 1; line += 1; column = 1; continue
        if char == "%":
            end = text.find("\n", index)
            end = length if end < 0 else end
            if not text[text.rfind("\n", 0, index) + 1:index].strip() and text[index:end].strip() == "%{": block = 1
            index = end; continue
        if text.startswith("...", index):
            end = text.find("\n", index)
            if end < 0: index = length
            else: index = end + 1; line += 1; column = 1
            continue
        previous = tokens[-1] if tokens else None
        ending = previous is not None and (previous[0] in {"id", "number"} or previous[1] in {")", "]", "}", "'", ".'"})
        adjacent = index > 0 and not text[index - 1].isspace()
        if char == "'" and ending and adjacent:
            tokens.append(("op", "'", line, column)); index += 1; column += 1; continue
        if text.startswith(".'", index) and ending:
            tokens.append(("op", ".'", line, column)); index += 2; column += 2; continue
        if char in {"'", '"'}:
            start_line, start_column, quote = line, column, char
            index += 1; column += 1
            while index < length:
                if text[index] == "\n": raise ContractError(f"Unterminated MATLAB string in {label}:{start_line}")
                if text[index] == quote:
                    if index + 1 < length and text[index + 1] == quote:
                        index += 2; column += 2; continue
                    index += 1; column += 1; break
                index += 1; column += 1
            else: raise ContractError(f"Unterminated MATLAB string in {label}:{start_line}")
            tokens.append(("string", "<literal>", start_line, start_column)); continue
        name = _NAME.match(text, index)
        if name:
            value = name.group(); tokens.append(("id", value, line, column)); index = name.end(); column += len(value); continue
        number = _NUMBER.match(text, index)
        if number:
            value = number.group(); tokens.append(("number", value, line, column)); index = number.end(); column += len(value); continue
        pair = text[index:index + 2]
        value = pair if pair in {"==", "~=", "<=", ">=", "&&", "||", ".*", "./", ".\\", ".^"} else char
        if value not in {"==", "~=", "<=", ">=", "&&", "||", ".*", "./", ".\\", ".^", "=", "<", ">", "&", "|", "~", "+", "-", "*", "/", "\\", "^", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}"}:
            raise ContractError(f"Unsupported MATLAB construct {value!r} in {label}:{line}; provide an explicit numeric thin adapter")
        tokens.append(("op", value, line, column)); index += len(value); column += len(value)
    if block: raise ContractError(f"Unterminated MATLAB block comment in {label}")
    return tokens


def _signature(tokens, start, label):
    stop = start + 1
    while stop < len(tokens) and tokens[stop][0] != "newline" and tokens[stop][1] != ";": stop += 1
    row = tokens[start + 1:stop]
    equals = [k for k, token in enumerate(row) if token[1] == "="]
    if len(equals) > 1 or not row: raise ContractError(f"Unsupported MATLAB function declaration in {label}:{tokens[start][2]}")
    if equals:
        split = equals[0]
        output_tokens, tail = row[:split], row[split + 1:]
        outputs = [token[1] for token in output_tokens if token[0] == "id"]
        if any(token[0] != "id" and token[1] not in {"[", "]", ",", "~"} for token in output_tokens):
            raise ContractError(f"Unsupported MATLAB output declaration in {label}")
    else: outputs, tail = [], row
    if not tail or tail[0][0] != "id": raise ContractError(f"An explicit named MATLAB function is required in {label}")
    name = tail[0][1]
    args = tail[1:]
    if args:
        if args[0][1] != "(" or args[-1][1] != ")" or any(token[0] != "id" and token[1] not in {",", "~"} for token in args[1:-1]):
            raise ContractError(f"Unsupported MATLAB input declaration in {label}")
        inputs = [token[1] for token in args[1:-1] if token[0] == "id" or token[1] == "~"]
    else: inputs = []
    if any(value in {"varargin", "varargout"} for value in inputs + outputs):
        raise ContractError("Variable-arity MATLAB source requires an explicit fixed-interface thin adapter")
    return {"name": name, "line": tokens[start][2], "inputs": inputs, "outputs": outputs}, stop


def _inspect(text, relative):
    tokens = _tokens(text, relative)
    visible = [token for token in tokens if token[0] != "newline"]
    if not visible or visible[0][1] != "function":
        raise ContractError(f"MATLAB snapshot .m file must be a function file, not a script or class: {relative}")
    functions, header_indices, headers = [], set(), {}
    for index, token in enumerate(tokens):
        if token[1] == "function":
            function, stop = _signature(tokens, index, relative)
            functions.append(function); header_indices.update(range(index, stop))
            headers[index] = (function, stop)
    if len({row["name"] for row in functions}) != len(functions):
        raise ContractError(f"Duplicate MATLAB function declarations in {relative}")
    if functions[0]["name"] != PurePosixPath(relative).stem:
        raise ContractError(f"MATLAB primary function must match its snapshot filename: {relative}")
    if any(row["name"].lower() in _FORBIDDEN for row in functions):
        raise ContractError("MATLAB source functions cannot shadow dynamic workspace/process/loading primitives")
    # Keep each function's variables separate.  A local helper assigning a name
    # must not make an unresolved call with that name in the main function look
    # like array indexing. Nested closures are deliberately outside this subset.
    variables = {row["name"]: set(row["inputs"] + row["outputs"]) for row in functions}
    scopes, blocks, delimiters, current = {}, [], [], None
    for index, token in enumerate(tokens):
        if index in headers:
            if blocks or delimiters:
                raise ContractError("Nested MATLAB functions/implicit multi-function endings require a flat explicit numeric thin adapter")
            current = headers[index][0]["name"]; blocks.append("function")
        if index in header_indices: continue
        scopes[index] = current
        value = token[1]
        if token[0] == "newline" or value == ";": continue
        if current is None:
            raise ContractError(f"MATLAB script statements outside function bodies are unsupported: {relative}:{token[2]}")
        if value in {"(", "[", "{"}: delimiters.append(value)
        elif value in {")", "]", "}"}:
            if not delimiters or delimiters.pop() != {")": "(", "]": "[", "}": "{"}[value]:
                raise ContractError(f"Unbalanced MATLAB delimiters in {relative}:{token[2]}")
        elif not delimiters and token[0] == "id" and not (index and tokens[index - 1][1] == "."):
            if value in {"if", "for", "while", "switch", "try", "arguments"}: blocks.append(value)
            elif value == "end":
                if not blocks: raise ContractError(f"Unbalanced MATLAB end in {relative}:{token[2]}")
                if blocks.pop() == "function": current = None
    if delimiters or (blocks and (blocks != ["function"] or len(functions) != 1)):
        raise ContractError(f"Unclosed MATLAB block/function in {relative}")
    # Collect local assignment roots, including indexed/struct assignments,
    # bracket multiple outputs, and loop/catch variables.  Scope is conservative
    # across the file; it never grants a callable external dependency.
    for index, token in enumerate(tokens):
        if index in header_indices: continue
        scope = scopes.get(index)
        if scope is None: continue
        names = variables[scope]
        if token[1] == "catch" and index + 1 < len(tokens) and tokens[index + 1][0] == "id": names.add(tokens[index + 1][1])
        if token[1] != "=": continue
        begin = index - 1
        while begin >= 0 and begin not in header_indices and tokens[begin][0] != "newline" and tokens[begin][1] != ";": begin -= 1
        left = tokens[begin + 1:index]
        if left and left[0][1] == "for": left = left[1:]
        if left and left[0][0] == "id": names.add(left[0][1])
        elif left and left[0][1] == "[": names.update(row[1] for row in left if row[0] == "id")
    return tokens, functions, variables, header_indices, scopes


def validate_matlab_sources(root, spec, expected_files, *, signature=(5, 1), extra_numeric_builtins=()):
    """Review every bound .m file without executing it; raise on unsupported use.

    ``root`` is the material artifact root and ``expected_files`` is the trusted
    artifact ``files`` relative-path/SHA256 map, not a user asserted flag.  Hash
    checks concern the same bytes lexed below.  The adapter must hash its private
    execution copy again; this static report is not execution authorization.
    """
    # These are trusted caller contracts, never fields read from source JSON.
    # The extension is deliberately closed: it cannot grant IO/dynamic calls.
    if (not isinstance(signature, tuple) or len(signature) != 2 or
            any(type(value) is not int for value in signature) or signature not in {(5, 1), (6, 3)}):
        raise ContractError("Only trusted plant (5,1) or external-expert (6,3) function contracts are supported")
    if (not isinstance(extra_numeric_builtins, (tuple, list, set, frozenset)) or
            any(not isinstance(name, str) or name not in _EXPERT_NUMERIC for name in extra_numeric_builtins)):
        raise ContractError("Trusted numeric extension is limited to quadprog/optimoptions and sparse matrices")
    numeric_builtins = _PURE | set(extra_numeric_builtins)
    root = Path(root).resolve(strict=True)
    if not isinstance(spec, dict) or spec.get("backend") not in {"matlab_function", "simulink"} or not isinstance(expected_files, dict):
        raise ContractError("Explicit MATLAB adapter and trusted material file hash map required")
    inventory = {}
    for name, digest in expected_files.items():
        relative = _relative(name)
        if relative in inventory or not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ContractError("MATLAB material requires unique normalized paths and SHA256 hashes")
        inventory[relative] = digest.lower()
    source_file = _relative(spec.get("source_file"))
    if source_file not in inventory: raise ContractError("MATLAB source is outside its bound material closure")
    for relative in inventory:
        suffix = PurePosixPath(relative).suffix.lower()
        if suffix in _OPAQUE or suffix.startswith(".mex") or any(part.startswith("@") for part in PurePosixPath(relative).parts):
            raise ContractError("Opaque MEX/P-code/class dependencies require an explicit supported thin adapter")
    actual_m = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.suffix.lower() == ".m" and path.is_file()}
    expected_m = {name for name in inventory if PurePosixPath(name).suffix.lower() == ".m"}
    if actual_m != expected_m: raise ContractError("MATLAB .m source file inventory differs from its bound material closure")
    if len(expected_m) > 256: raise ContractError("MATLAB static source budget exceeds 256 function files")
    records, parsed, total = [], {}, 0
    for relative in sorted(expected_m):
        path = _checked(root, relative)
        payload = path.read_bytes(); total += len(payload)
        if len(payload) > 1024 * 1024 or total > 8 * 1024 * 1024:
            raise ContractError("MATLAB static source budget exceeds 1 MiB/file or 8 MiB total")
        if hashlib.sha256(payload).hexdigest() != inventory[relative]:
            raise ContractError(f"MATLAB source hash changed before static review/execution: {relative}")
        try: text = payload.decode("utf-8-sig")
        except UnicodeError as error: raise ContractError("MATLAB source subset requires UTF-8 function files") from error
        tokens, functions, variables, headers, scopes = _inspect(text, relative)
        parsed[relative] = (tokens, functions, variables, headers, scopes)
        package = [part[1:] for part in PurePosixPath(relative).parent.parts if part.startswith("+")]
        qualified = ".".join([*package, functions[0]["name"]])
        records.append({"path": relative, "sha256": inventory[relative], "primary_function": qualified, "functions": functions})
    exported = {row["primary_function"]: row for row in records}
    if len(exported) != len(records): raise ContractError("Ambiguous MATLAB primary function names require a smaller explicit dependency closure")
    if set(exported) & set(extra_numeric_builtins):
        raise ContractError("External expert sources cannot shadow the declared installed numerical solvers")
    function_root = PurePosixPath(source_file).parent
    if spec["backend"] == "matlab_function":
        entrypoint = spec.get("entrypoint")
        if not isinstance(entrypoint, str) or not all(_NAME.fullmatch(part) for part in entrypoint.split(".")):
            raise ContractError("An explicit MATLAB entrypoint is required")
        for _ in entrypoint.split(".")[:-1]: function_root = function_root.parent
    for record in records:
        relative = record["path"]
        tokens, functions, variables, headers, scopes = parsed[relative]
        local_functions = {row["name"] for row in functions}
        dependencies, builtins = set(), set()
        for index, token in enumerate(tokens):
            if token[0] != "id" or index in headers: continue
            if index and tokens[index - 1][1] == ".": continue  # a struct field or qualified suffix
            name = token[1]
            if name.lower() in _FORBIDDEN:
                raise ContractError(f"Unsupported MATLAB hidden-state/workspace/process/loading construct {name} in {relative}:{token[2]}; provide an explicit numeric thin adapter")
            if name in variables.get(scopes.get(index), set()) or name in local_functions or name in _KEYWORDS: continue
            # Qualified package calls must resolve within the frozen inventory.
            end, parts = index, [name]
            while end + 2 < len(tokens) and tokens[end + 1][1] == "." and tokens[end + 2][0] == "id":
                parts.append(tokens[end + 2][1]); end += 2
            qualified = ".".join(parts)
            if qualified in exported:
                target = PurePosixPath(exported[qualified]["path"])
                public = function_root.joinpath(*[f"+{part}" for part in parts[:-1]], parts[-1] + ".m")
                private = PurePosixPath(relative).parent / "private" / (name + ".m")
                if target == public or (len(parts) == 1 and target == private):
                    dependencies.add(qualified); continue
                raise ContractError(f"MATLAB helper {qualified} is outside the explicit function path/private dependency scope; provide a thin adapter")
            if len(parts) == 1 and name in numeric_builtins:
                builtins.add(name); continue
            raise ContractError(f"Unknown/external MATLAB dependency {qualified} in {relative}:{token[2]}; include a supported snapshot helper or provide an explicit numeric thin adapter")
        record["snapshot_dependencies"] = sorted(dependencies)
        record["numerical_builtins"] = sorted(builtins)
    if spec["backend"] == "matlab_function":
        entry = exported.get(spec.get("entrypoint"))
        if entry is None or entry["path"] != source_file:
            raise ContractError("MATLAB entrypoint must resolve to the declared bound function file")
        primary = entry["functions"][0]
        if len(primary["inputs"]) != signature[0] or len(primary["outputs"]) != signature[1]:
            raise ContractError("MATLAB function requires exactly %s inputs and %s outputs" % signature)
        if "output_entrypoint" in spec:
            output = exported.get(spec["output_entrypoint"])
            if output is None or len(output["functions"][0]["inputs"]) != 2 or len(output["functions"][0]["outputs"]) != 1:
                raise ContractError("MATLAB output requires a bound g(x,parameters) function with one output")
    return {"schema_version": 1, "status": "supported_subset", "scope": "explicit_numeric_function_files",
            "source_executed": False, "files": records, "function_inventory": sorted(exported),
            "entrypoint_signature": list(signature), "extra_numeric_builtins": sorted(set(extra_numeric_builtins)),
            "file_count": len(records), "reviewed_bytes": total,
            "limitations": ["Static support checks are not a MATLAB parser or security sandbox.",
                "Installed numerical builtin resolution remains a trusted MATLAB environment contract.",
                "Only flat function files, local functions, explicit packages and reachable private helpers are supported; nested closures are rejected.",
                "Transpose operators must use unambiguous adjacent A' or A.' notation; ambiguous whitespace forms require a thin wrapper.",
                "Unknown external functions, hidden state and dynamic execution require an explicit supported thin adapter.",
                "Repeated-input checks cover visited inputs and do not formally prove purity or physical fidelity."]}
