; ============================================================
; Python — structural relationships
; Capture names used by parser.py:
;   call.callee     — function/method being called
;   import.module   — module being imported
;   import.name     — specific name from a from-import
;   inherit.child   — subclass name
;   inherit.parent  — superclass name
;   type.use        — type annotation reference
; ============================================================

; --- CALLS ---

; Direct call: foo()
(call
  function: (identifier) @call.callee)

; Method call: obj.method()
(call
  function: (attribute
    attribute: (identifier) @call.callee))

; --- IMPORTS ---

; import os
; import os.path
(import_statement
  name: (dotted_name) @import.module)

; import os as o
(import_statement
  name: (aliased_import
    name: (dotted_name) @import.module))

; from os import path
(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.name)

; from os import path as p
(import_from_statement
  module_name: (dotted_name) @import.module
  name: (aliased_import
    name: (dotted_name) @import.name))


; from . import something  (relative import, module is empty)
(import_from_statement
  module_name: (relative_import) @import.module)

; from os import *
; wildcard_import is an unnamed child — field label "name:" is not valid here
(import_from_statement
  module_name: (dotted_name) @import.module
  (wildcard_import) @import.name)

; --- INHERITANCE ---

; class Foo(Bar):
(class_definition
  name: (identifier) @inherit.child
  superclasses: (argument_list
    (identifier) @inherit.parent))

; class Foo(pkg.Bar):
(class_definition
  name: (identifier) @inherit.child
  superclasses: (argument_list
    (attribute) @inherit.parent))

; --- TYPE ANNOTATIONS ---

; def foo(x: Bar) -> Baz:  — return type
(function_definition
  name: (identifier) @_fn
  return_type: (type
    (identifier) @type.use))

; def foo(x: SomeType):  — parameter type
(parameters
  (typed_parameter
    type: (type
      (identifier) @type.use)))

; variable annotation: x: Foo = ...
(expression_statement
  (assignment
    left: (identifier)
    type: (type
      (identifier) @type.use)))
