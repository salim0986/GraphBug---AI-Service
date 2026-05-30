; ============================================================
; JavaScript — structural relationships
; ============================================================

; --- CALLS ---

(call_expression
  function: (identifier) @call.callee)

(call_expression
  function: (member_expression
    property: (property_identifier) @call.callee))

(new_expression
  constructor: (identifier) @call.callee)

(new_expression
  constructor: (member_expression
    property: (property_identifier) @call.callee))

; --- IMPORTS ---

; import ... from 'module'
(import_statement
  source: (string
    (string_fragment) @import.module))

; import { foo } — capture specific names
(import_specifier
  name: (identifier) @import.name)

; require('module')
(call_expression
  function: (identifier) @_req
  (#eq? @_req "require")
  arguments: (arguments
    (string
      (string_fragment) @import.module)))

; --- INHERITANCE ---

; class Foo extends Bar
(class_declaration
  name: (identifier) @inherit.child
  (class_heritage
    (identifier) @inherit.parent))

(class_declaration
  name: (identifier) @inherit.child
  (class_heritage
    (member_expression
      property: (property_identifier) @inherit.parent)))
