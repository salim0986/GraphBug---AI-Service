; ============================================================
; TypeScript — structural relationships
; ============================================================

; --- CALLS ---

; foo()
(call_expression
  function: (identifier) @call.callee)

; obj.method()
(call_expression
  function: (member_expression
    property: (property_identifier) @call.callee))

; new Foo()
(new_expression
  constructor: (identifier) @call.callee)

; new pkg.Foo()
(new_expression
  constructor: (member_expression
    property: (property_identifier) @call.callee))

; --- IMPORTS ---

; import ... from 'module'
(import_statement
  source: (string
    (string_fragment) @import.module))

; import { foo } from 'module'  — capture each named specifier
(import_specifier
  name: (identifier) @import.name)

; import * as ns from 'module'  — namespace import
(namespace_import
  (identifier) @import.name)

; --- INHERITANCE ---

; class Foo extends Bar
(class_declaration
  name: (type_identifier) @inherit.child
  (class_heritage
    (extends_clause
      value: (identifier) @inherit.parent)))

; class Foo extends ns.Bar
(class_declaration
  name: (type_identifier) @inherit.child
  (class_heritage
    (extends_clause
      value: (member_expression
        property: (property_identifier) @inherit.parent))))

; class Foo implements Bar, Baz
(class_declaration
  name: (type_identifier) @implement.child
  (class_heritage
    (implements_clause
      (type_identifier) @implement.interface)))

; interface IFoo extends IBar
(interface_declaration
  name: (type_identifier) @inherit.child
  (extends_type_clause
    type: (type_identifier) @inherit.parent))

; --- TYPE USES ---

; function return type:  ): ReturnType
(type_annotation
  (type_identifier) @type.use)

; parameter type:  (param: SomeType)
(required_parameter
  type: (type_annotation
    (type_identifier) @type.use))

(optional_parameter
  type: (type_annotation
    (type_identifier) @type.use))
