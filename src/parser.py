"""
Universal Parser — M1 upgrade
Extracts declarations AND structural relationships (calls, imports, inheritance, type-use)
from source files using tree-sitter AST queries.

Backward-compatible: parse_file() / parse_code() still return (captures, code_bytes)
New: extract_relationships() returns a ParseReport with typed records.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from tree_sitter_languages import get_language, get_parser
import tree_sitter
from .logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Language → file-extension map (unchanged from original)
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    # Web / JS
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    # Backend / Core
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    # Systems / Scripting
    ".sh": "bash",
    ".lua": "lua",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    # Config / Infra
    ".tf": "hcl",
    ".dockerfile": "dockerfile",
    ".make": "make",
    # Docs
    ".md": "markdown",
}


def detect_language_from_filename(filename: str) -> Optional[str]:
    """Return language name from file extension, or None if unsupported."""
    _, ext = os.path.splitext(filename)
    return EXTENSION_MAP.get(ext.lower())


# ---------------------------------------------------------------------------
# Typed relationship records
# ---------------------------------------------------------------------------

@dataclass
class Declaration:
    """A named declaration (function, class, method, interface, struct …)."""
    name: str
    type: str          # "function" | "class" | "interface" | "method" | "struct" | "other"
    file: str
    start_line: int
    end_line: int
    code: str = ""     # first 2000 chars of the declaration body


@dataclass
class Call:
    """A function/method call site."""
    caller: str        # enclosing function name, or "module_level"
    callee: str        # name of the function/method being called
    file: str
    line: int


@dataclass
class Import:
    """An import / include / use statement."""
    file: str
    module: str        # module or path being imported
    alias: Optional[str]  # specific name imported (e.g. `from os import path`)
    line: int


@dataclass
class Inheritance:
    """A class-extends or class-implements relationship."""
    child: str         # the subclass / implementing class
    parent: str        # the superclass or interface
    file: str
    line: int


@dataclass
class TypeUse:
    """A type annotation reference inside a function or class."""
    context: str       # enclosing function / class name
    type_name: str
    file: str
    line: int


@dataclass
class ParseReport:
    """Full extraction result for one file."""
    file: str
    language: Optional[str]
    status: str = "failed"   # "success" | "partial" | "failed"
    declarations: List[Declaration] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    imports: List[Import] = field(default_factory=list)
    inheritances: List[Inheritance] = field(default_factory=list)
    type_uses: List[TypeUse] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def nodes_extracted(self) -> int:
        return len(self.declarations)

    @property
    def relationships_extracted(self) -> int:
        return len(self.calls) + len(self.imports) + len(self.inheritances)


# ---------------------------------------------------------------------------
# Node-type → declaration-type mapping
# ---------------------------------------------------------------------------

_DECL_TYPE_MAP = {
    "function_definition":   "function",
    "function_declaration":  "function",
    "function_item":         "function",
    "method_definition":     "method",
    "method_declaration":    "method",
    "class_definition":      "class",
    "class_declaration":     "class",
    "class_specifier":       "class",
    "struct_item":           "struct",
    "struct_specifier":      "struct",
    "interface_declaration": "interface",
    "trait_item":            "interface",
    "impl_item":             "class",
    "enum_declaration":      "class",
    "enum_item":             "class",
    "abstract_class_declaration": "class",
    "namespace_declaration": "other",
    "type_alias_declaration": "other",
}

# Node types that represent a function/method scope (for enclosing-name lookup)
_ENCLOSING_FUNC_TYPES = frozenset({
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "arrow_function",
    "function_expression",
    "closure_expression",
    "lambda",
})

# Class-scope node types across supported languages.
# Used to detect when a function_definition is actually a method.
_CLASS_SCOPE_TYPES = frozenset({
    "class_definition",    # Python
    "class_declaration",   # JS / TS / Java / C#
    "class_specifier",     # C++
    "impl_item",           # Rust
    "class_body",          # Java (body of a class)
    "declaration_list",    # C# class body
})


def _is_inside_class(func_node: Any) -> bool:
    """
    Return True when *func_node* is a method — i.e. its nearest enclosing
    structural scope is a class, not another function.

    Stops walking upward the moment it hits either:
    - a class-scope node  → True (it's a method)
    - another function node → False (it's a nested function, not a method)
    """
    current = func_node.parent
    while current is not None:
        t = current.type
        if t in _CLASS_SCOPE_TYPES:
            return True
        if t in _ENCLOSING_FUNC_TYPES:
            return False  # Crossed a function boundary before any class
        current = current.parent
    return False


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class UniversalParser:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_query_path = os.path.join(current_dir, "queries")

    # ------------------------------------------------------------------
    # BACKWARD-COMPATIBLE API
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> Tuple[Optional[Any], Optional[bytes]]:
        """
        Original interface: (captures, code_bytes) or (None, None).
        Still used by graph_builder.process_file_nodes().
        """
        _, ext = os.path.splitext(file_path)
        lang_name = EXTENSION_MAP.get(ext)
        if not lang_name:
            return None, None
        try:
            language = get_language(lang_name)
            parser = get_parser(lang_name)
            with open(file_path, "rb") as f:
                code_bytes = f.read()
            tree = parser.parse(code_bytes)
            query_file = os.path.join(self.base_query_path, lang_name, "tags.scm")
            if not os.path.exists(query_file):
                return None, None
            with open(query_file) as f:
                query_scm = f.read()
            query = language.query(query_scm)
            captures = query.captures(tree.root_node)
            return captures, code_bytes
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None, None
        except Exception as e:
            logger.warning(f"Parser error in {file_path}: {e}")
            return None, None

    def parse_file_ast(self, file_path: str) -> Tuple[Optional[Any], Optional[bytes], Optional[str]]:
        """
        Returns (tree, code_bytes, language_name).
        Used by the new SemanticChunker in M2.
        """
        _, ext = os.path.splitext(file_path)
        lang_name = EXTENSION_MAP.get(ext)
        if not lang_name:
            return None, None, None
        try:
            parser = get_parser(lang_name)
            with open(file_path, "rb") as f:
                code_bytes = f.read()
            tree = parser.parse(code_bytes)
            return tree, code_bytes, lang_name
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None, None, None
        except Exception as e:
            logger.warning(f"Parser error in {file_path}: {e}")
            return None, None, None

    def parse_code(self, code_bytes: bytes, lang_name: str) -> Tuple[Optional[Any], Optional[bytes]]:
        """
        Original interface: (captures, code_bytes) or (None, None).
        """
        if not lang_name or lang_name == "text":
            return None, None
        try:
            language = get_language(lang_name)
            parser = get_parser(lang_name)
            tree = parser.parse(code_bytes)
            query_file = os.path.join(self.base_query_path, lang_name, "tags.scm")
            if not os.path.exists(query_file):
                return None, None
            with open(query_file) as f:
                query_scm = f.read()
            query = language.query(query_scm)
            captures = query.captures(tree.root_node)
            return captures, code_bytes
        except Exception as e:
            logger.warning(f"Parser error for {lang_name}: {e}")
            return None, None

    # ------------------------------------------------------------------
    # NEW M1 API — relationship extraction
    # ------------------------------------------------------------------

    def extract_relationships(
        self,
        code_bytes: bytes,
        lang_name: str,
        file_path: str = "",
    ) -> ParseReport:
        """
        Extract declarations + structural relationships from code bytes.

        Returns a ParseReport with:
          - declarations  (functions, classes, interfaces …)
          - calls         (caller → callee)
          - imports       (module dependencies)
          - inheritances  (extends / implements)
          - type_uses     (type annotations)
          - status        success | partial | failed
          - errors        human-readable error list
        """
        report = ParseReport(file=file_path, language=lang_name)

        if not lang_name or lang_name == "text":
            report.errors.append(f"Unsupported language: {lang_name!r}")
            return report

        try:
            language = get_language(lang_name)
            ts_parser = get_parser(lang_name)
            tree = ts_parser.parse(code_bytes)
        except Exception as e:
            report.errors.append(f"Tree-sitter init error: {e}")
            return report

        # --- Step 1: declarations from tags.scm ---
        tags_ok = self._extract_declarations(tree, language, code_bytes, file_path, report)

        # --- Step 2: relationships from relationships.scm ---
        rels_ok = self._extract_rels(tree, language, code_bytes, file_path, report)

        # --- Step 3: detect AST syntax errors ---
        # tree-sitter is error-tolerant; it inserts ERROR/MISSING nodes for invalid
        # syntax rather than refusing to parse.  These nodes produce garbage captures.
        has_syntax_errors = tree.root_node.has_error
        if has_syntax_errors:
            report.errors.append("Source contains syntax errors — AST ERROR nodes present; results may be incomplete")

        if tags_ok and rels_ok and not has_syntax_errors:
            report.status = "success"
        elif tags_ok or rels_ok:
            report.status = "partial"
        else:
            report.status = "failed"

        logger.debug(
            f"[Parser] {file_path} ({lang_name}): "
            f"status={report.status} decls={report.nodes_extracted} "
            f"calls={len(report.calls)} imports={len(report.imports)} "
            f"inherit={len(report.inheritances)}"
        )
        return report

    def extract_relationships_from_file(self, file_path: str) -> ParseReport:
        """Convenience wrapper: read file then call extract_relationships()."""
        _, ext = os.path.splitext(file_path)
        lang_name = EXTENSION_MAP.get(ext.lower())
        report = ParseReport(file=file_path, language=lang_name)

        if not lang_name:
            report.errors.append(f"Unsupported extension: {ext!r}")
            return report
        try:
            with open(file_path, "rb") as f:
                code_bytes = f.read()
            return self.extract_relationships(code_bytes, lang_name, file_path)
        except FileNotFoundError:
            report.errors.append(f"File not found: {file_path}")
            return report
        except Exception as e:
            report.errors.append(f"Read error: {e}")
            return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_declarations(
        self,
        tree: Any,
        language: Any,
        code_bytes: bytes,
        file_path: str,
        report: ParseReport,
    ) -> bool:
        """Populate report.declarations from tags.scm."""
        tags_file = os.path.join(self.base_query_path, language.__class__.__name__.lower(), "tags.scm")
        # language object doesn't expose name directly; use a workaround
        lang_name = report.language or ""
        tags_file = os.path.join(self.base_query_path, lang_name, "tags.scm")

        if not os.path.exists(tags_file):
            report.errors.append(f"No tags.scm for {lang_name}")
            return False
        try:
            with open(tags_file) as f:
                tags_scm = f.read()
            query = language.query(tags_scm)
            captures = query.captures(tree.root_node)
        except Exception as e:
            report.errors.append(f"tags.scm compile/run error: {e}")
            return False

        seen: set = set()
        for node, cap_name in captures:
            if cap_name != "name":
                continue
            parent = node.parent
            if parent is None:
                continue
            parent_type = parent.type
            # Avoid duplicate decls (same start byte)
            key = (parent.start_byte, parent_type)
            if key in seen:
                continue
            seen.add(key)

            decl_type = _DECL_TYPE_MAP.get(parent_type, "other")
            # Promote function_definition → "method" when the nearest enclosing
            # structural scope is a class.  Python methods are syntactically
            # function_definitions; only context distinguishes them.
            if decl_type == "function" and _is_inside_class(parent):
                decl_type = "method"
            name_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            body = code_bytes[parent.start_byte:parent.end_byte].decode("utf-8", errors="ignore")[:2000]

            report.declarations.append(Declaration(
                name=name_text,
                type=decl_type,
                file=file_path,
                start_line=parent.start_point[0],
                end_line=parent.end_point[0],
                code=body,
            ))
        return True

    def _extract_rels(
        self,
        tree: Any,
        language: Any,
        code_bytes: bytes,
        file_path: str,
        report: ParseReport,
    ) -> bool:
        """Populate calls/imports/inheritances/type_uses from relationships.scm."""
        lang_name = report.language or ""
        rels_file = os.path.join(self.base_query_path, lang_name, "relationships.scm")

        if not os.path.exists(rels_file):
            report.errors.append(f"No relationships.scm for {lang_name}")
            return False
        try:
            with open(rels_file) as f:
                rels_scm = f.read()
            query = language.query(rels_scm)
            captures = query.captures(tree.root_node)
        except Exception as e:
            report.errors.append(f"relationships.scm error: {e}")
            return False

        # Process captures in document order (tree-sitter guarantees this)
        # We track "current inherit child" while walking the list so that
        # subsequent inherit.parent captures can reference the right child.
        current_inherit_child: Optional[Tuple[str, int]] = None  # (name, line)
        current_implement_child: Optional[Tuple[str, int]] = None
        current_implement_interface: Optional[Tuple[str, int]] = None
        pending_import_module: Optional[Tuple[str, int]] = None  # (module_text, line)
        # Dedup set: prevent the same call expression from being recorded twice
        # when multiple SCM patterns match the same AST node.
        seen_calls: set = set()

        for node, cap_name in captures:
            text = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()
            line = node.start_point[0]

            if cap_name == "call.callee":
                caller = self._find_enclosing_name(node, code_bytes)
                if text:
                    call_key = (caller, text, line)
                    if call_key not in seen_calls:
                        seen_calls.add(call_key)
                        report.calls.append(Call(caller=caller, callee=text, file=file_path, line=line))

            elif cap_name == "import.module":
                # Strip surrounding quotes (string literals in some languages)
                module = text.strip("\"'`")
                pending_import_module = (module, line)
                report.imports.append(Import(file=file_path, module=module, alias=None, line=line))

            elif cap_name == "import.name":
                name_clean = text.strip("\"'`")
                if pending_import_module and report.imports:
                    # Attach the specific name to the most recent import
                    last = report.imports[-1]
                    if last.alias is None:
                        report.imports[-1] = Import(
                            file=last.file,
                            module=last.module,
                            alias=name_clean,
                            line=last.line,
                        )
                else:
                    report.imports.append(Import(file=file_path, module=name_clean, alias=None, line=line))

            elif cap_name == "inherit.child":
                current_inherit_child = (text, line)

            elif cap_name == "inherit.parent":
                if current_inherit_child and text:
                    child_name, child_line = current_inherit_child
                    report.inheritances.append(Inheritance(
                        child=child_name,
                        parent=text,
                        file=file_path,
                        line=child_line,
                    ))

            elif cap_name == "implement.child":
                current_implement_child = (text, line)
                if current_implement_interface:
                    interface_name, _ = current_implement_interface
                    report.inheritances.append(Inheritance(
                        child=text,
                        parent=interface_name,
                        file=file_path,
                        line=line,
                    ))
                    current_implement_child = None
                    current_implement_interface = None

            elif cap_name == "implement.interface":
                current_implement_interface = (text, line)
                if current_implement_child and text:
                    child_name, child_line = current_implement_child
                    report.inheritances.append(Inheritance(
                        child=child_name,
                        parent=text,
                        file=file_path,
                        line=child_line,
                    ))
                    current_implement_child = None
                    current_implement_interface = None

            elif cap_name == "type.use":
                context = self._find_enclosing_name(node, code_bytes)
                if text and text not in ("None", "void", "bool", "int", "str", "float"):
                    report.type_uses.append(TypeUse(
                        context=context,
                        type_name=text,
                        file=file_path,
                        line=line,
                    ))

        return True

    def _find_enclosing_name(self, node: Any, code_bytes: bytes) -> str:
        """
        Walk up the AST to find the nearest enclosing function/method name.
        Returns "module_level" if no enclosing function is found.
        """
        current = node.parent
        while current is not None:
            if current.type in _ENCLOSING_FUNC_TYPES:
                name_node = current.child_by_field_name("name")
                if name_node:
                    return code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            current = current.parent
        return "module_level"
