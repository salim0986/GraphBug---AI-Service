; ============================================================
; Rust — structural relationships
; ============================================================

; --- CALLS ---

; foo()  /  foo::<T>()
(call_expression
  function: (identifier) @call.callee)

; self.method()  /  obj.method()
(call_expression
  function: (field_expression
    field: (field_identifier) @call.callee))

; path::to::func()
(call_expression
  function: (scoped_identifier
    name: (identifier) @call.callee))

; Macro invocations: vec![], println![]
(macro_invocation
  macro: (identifier) @call.callee)

; --- IMPORTS ---

; use std::io::Write;
(use_declaration
  argument: (scoped_identifier
    name: (identifier) @import.module))

; use std::io::{Read, Write};
(use_declaration
  argument: (scoped_use_list
    path: (identifier) @import.module))

; use std::io::*;
(use_declaration
  argument: (use_wildcard
    (identifier) @import.module))

; use foo as bar;
(use_as_clause
  path: (identifier) @import.module
  alias: (identifier) @import.name)

; --- INHERITANCE (trait implementations) ---

; impl Trait for Struct  →  Struct implements Trait
(impl_item
  trait: (type_identifier) @implement.interface
  type: (type_identifier) @implement.child)

; Also generic impls: impl<T> Trait for Struct<T>
(impl_item
  trait: (generic_type
    type: (type_identifier) @implement.interface)
  type: (type_identifier) @implement.child)
