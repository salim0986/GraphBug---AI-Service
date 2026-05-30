import tiktoken
from typing import List, Dict, Any, Optional
from tree_sitter import Node
import os

class SemanticChunker:
    """
    Splits source code into semantic chunks respecting AST boundaries.
    Uses tiktoken for fast token estimation.
    """
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Use cl100k_base as a fast generic tokenizer approximation
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # AST node types that represent structural boundaries
        self.function_types = {
            "function_definition", "function_declaration", "function_item",
            "method_definition", "method_declaration",
            "arrow_function", "generator_function_declaration"
        }
        self.class_types = {
            "class_definition", "class_declaration",
            "interface_declaration", "impl_item"
        }

    def _get_tokens(self, text: str) -> List[int]:
        return self.encoder.encode(text, disallowed_special=())
        
    def _decode_tokens(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    def _split_large_text(self, text: str, start_line: int, parent_func: Optional[str], parent_class: Optional[str]) -> List[Dict[str, Any]]:
        """Split a large text block into overlapping chunks."""
        tokens = self._get_tokens(text)
        chunks = []

        if len(tokens) <= self.max_tokens:
            return [{
                "text": text,
                "start_line": start_line,
                "end_line": start_line + text.count("\n"),
                "parent_function": parent_func,
                "parent_class": parent_class
            }]

        step = self.max_tokens - self.overlap_tokens
        if step <= 0:
            step = self.max_tokens  # Fallback: no overlap

        # Pre-compute cumulative newline counts at each token boundary.
        # This turns line-number estimation from O(N²) into O(N) total.
        # We walk the text once, mapping token offset → newline count.
        token_newlines: List[int] = [0] * (len(tokens) + 1)
        decoded_so_far = ""
        for idx in range(len(tokens)):
            decoded_so_far = self._decode_tokens(tokens[:idx + 1])
            token_newlines[idx + 1] = decoded_so_far.count("\n")

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self._decode_tokens(chunk_tokens)
            chunk_start_line = start_line + token_newlines[i]
            chunk_end_line = start_line + token_newlines[min(i + self.max_tokens, len(tokens))]

            chunks.append({
                "text": chunk_text,
                "start_line": chunk_start_line,
                "end_line": chunk_end_line,
                "parent_function": parent_func,
                "parent_class": parent_class
            })

        return chunks

    def _extract_name(self, node: Node, code_bytes: bytes) -> Optional[str]:
        """Extract the name of a function or class node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return code_bytes[name_node.start_byte:name_node.end_byte].decode("utf8", errors="ignore")
        
        # Fallback: find the first identifier child
        for child in node.children:
            if child.type == "identifier" or child.type == "type_identifier":
                return code_bytes[child.start_byte:child.end_byte].decode("utf8", errors="ignore")
        return None

    def _process_node(
        self, 
        node: Node, 
        code_bytes: bytes, 
        current_class: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Recursively process nodes, extracting semantic chunks."""
        chunks = []
        node_type = node.type
        
        if node_type in self.class_types:
            # We found a class, update current_class and process children
            class_name = self._extract_name(node, code_bytes) or current_class
            node_text = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            tokens = self._get_tokens(node_text)
            if len(tokens) <= self.max_tokens:
                chunks.append({
                    "text": node_text,
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "parent_function": None,
                    "parent_class": class_name
                })
            else:
                for child in node.children:
                    chunks.extend(self._process_node(child, code_bytes, class_name))
            return chunks

        elif node_type in self.function_types:
            # We found a function/method. Treat it as a single unit (split if too large)
            func_name = self._extract_name(node, code_bytes)
            node_text = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            chunks.extend(self._split_large_text(
                text=node_text,
                start_line=node.start_point[0],
                parent_func=func_name,
                parent_class=current_class
            ))
            return chunks

        # If it's a root node, process its children
        if node.parent is None:
            for child in node.children:
                chunks.extend(self._process_node(child, code_bytes, current_class))
            return chunks
            
        # Top-level statements (e.g. imports) or class-level statements (e.g. fields)
        # We can identify them if their parent is the root or if their parent is a class/block
        # To avoid leaf nodes, we only chunk statements (nodes with children or specific types)
        # A simpler way: if we are here and we didn't recurse yet, we should recurse to find functions, 
        # but also capture non-function statements.
        # Actually, let's just recurse. For simplicity, we might lose some non-function statements if they are 
        # nested and we don't capture them. Let's capture them if they are direct children of root or class body.
        
        is_top_level = node.parent and node.parent.parent is None
        is_class_level = current_class and node.parent and node.parent.type in ("block", "declaration_list")
        
        if is_top_level or is_class_level:
            node_text = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            if node_text.strip() and not any(child.type in self.function_types for child in node.children):
                chunks.extend(self._split_large_text(
                    text=node_text,
                    start_line=node.start_point[0],
                    parent_func=None,
                    parent_class=current_class
                ))
                return chunks
                
        # Otherwise, keep recursing to find nested functions/classes
        for child in node.children:
            chunks.extend(self._process_node(child, code_bytes, current_class))
                
        return chunks

    def chunk_ast(self, tree: Any, code_bytes: bytes, file_path: str, language: str) -> List[Dict[str, Any]]:
        """
        Takes a tree-sitter AST and code, returns list of chunks.
        Each chunk: {file, start_line, end_line, parent_function, parent_class, language, raw_code}
        """
        raw_chunks = self._process_node(tree.root_node, code_bytes)
        
        # Post-process to add file and language metadata, and group tiny adjacent chunks?
        # Grouping tiny adjacent chunks (like sequential imports) makes retrieval better.
        grouped_chunks = []
        current_group = None
        current_tokens = 0
        
        for c in raw_chunks:
            c_tokens = len(self._get_tokens(c["text"]))
            
            # If the chunk has a function or class context, keep it separate from global blocks
            has_context = c["parent_function"] is not None or c["parent_class"] is not None
            
            if not has_context and current_group is not None and (current_tokens + c_tokens) <= self.max_tokens:
                # Append to current group
                current_group["text"] += "\n" + c["text"]
                current_group["end_line"] = c["end_line"]
                current_tokens += c_tokens
            else:
                if current_group is not None:
                    grouped_chunks.append(current_group)
                    
                if has_context or c_tokens > 20: # Only start new group if it's substantial or has context
                    current_group = c.copy()
                    current_tokens = c_tokens
                elif not has_context:
                    # Start a new group anyway for small global statements
                    current_group = c.copy()
                    current_tokens = c_tokens
                    
        if current_group is not None:
            grouped_chunks.append(current_group)
            
        # Finalize format
        final_chunks = []
        for c in grouped_chunks:
            final_chunks.append({
                "file": file_path,
                "language": language,
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "parent_function": c.get("parent_function"),
                "parent_class": c.get("parent_class"),
                "raw_code": c["text"].strip()
            })
            
        return final_chunks
