; ============================================================
; Java — structural relationships
; ============================================================

; --- CALLS ---

; method()  /  obj.method()
(method_invocation
  name: (identifier) @call.callee)

; new Foo()
(object_creation_expression
  type: (type_identifier) @call.callee)

; --- IMPORTS ---

; import java.util.List;
(import_declaration
  (scoped_identifier) @import.module)

; import static java.util.Arrays.*;
(import_declaration
  "static"
  (scoped_identifier) @import.module)

; --- INHERITANCE ---

; class Foo extends Bar
(class_declaration
  name: (identifier) @inherit.child
  (superclass
    (type_identifier) @inherit.parent))

; class Foo implements Bar, Baz
(class_declaration
  name: (identifier) @implement.child
  (super_interfaces
    (type_list
      (type_identifier) @implement.interface)))

; interface IFoo extends IBar
(interface_declaration
  name: (identifier) @inherit.child
  (extends_interfaces
    (type_list
      (type_identifier) @inherit.parent)))

; --- TYPE USES ---

; return type
(method_declaration
  type: (type_identifier) @type.use)

; variable declaration type
(local_variable_declaration
  type: (type_identifier) @type.use)
