; ============================================================
; C++ — structural relationships
; ============================================================

; --- CALLS ---

; foo()
(call_expression
  function: (identifier) @call.callee)

; obj.method()  /  ptr->method()
(call_expression
  function: (field_expression
    field: (field_identifier) @call.callee))

; ns::func()  /  Class::static_method()
(call_expression
  function: (qualified_identifier
    name: (identifier) @call.callee))

; --- IMPORTS (includes) ---

; #include "header.h"
(preproc_include
  path: (string_literal) @import.module)

; #include <vector>
(preproc_include
  path: (system_lib_string) @import.module)

; --- INHERITANCE ---

; class Foo : public Bar
(class_specifier
  name: (type_identifier) @inherit.child
  (base_class_clause
    (type_identifier) @inherit.parent))

; Multiple inheritance: class Foo : public Bar, public Baz
(class_specifier
  name: (type_identifier) @inherit.child
  (base_class_clause
    (qualified_identifier
      name: (identifier) @inherit.parent)))
