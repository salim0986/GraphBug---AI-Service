; ============================================================
; Go — structural relationships
; ============================================================

; --- CALLS ---

; foo()
(call_expression
  function: (identifier) @call.callee)

; pkg.Foo() or obj.Method()
(call_expression
  function: (selector_expression
    field: (field_identifier) @call.callee))

; --- IMPORTS ---

; import "fmt"
(import_spec
  path: (interpreted_string_literal) @import.module)

; import f "fmt"  (aliased)
(import_spec
  name: (package_identifier) @import.name
  path: (interpreted_string_literal) @import.module)

; --- INHERITANCE / INTERFACE EMBEDDING ---

; Interface embedding
(type_spec
  name: (type_identifier) @inherit.child
  type: (interface_type
    (constraint_elem
      [
        (type_identifier) @inherit.parent
        (qualified_type name: (type_identifier) @inherit.parent)
      ])))

; Struct embedding
(type_spec
  name: (type_identifier) @inherit.child
  type: (struct_type
    (field_declaration_list
      (field_declaration
        type: [
          (type_identifier) @inherit.parent
          (qualified_type name: (type_identifier) @inherit.parent)
        ]))))
