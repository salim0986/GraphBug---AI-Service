; ============================================================
; C# — structural relationships
; ============================================================

; --- CALLS ---

; Method() or obj.Method()
(invocation_expression
  function: (member_access_expression
    name: (identifier) @call.callee))

; Direct identifier call
(invocation_expression
  function: (identifier) @call.callee)

; new Foo()
(object_creation_expression
  type: (identifier) @call.callee)

; --- IMPORTS ---

; using System.Collections;
(using_directive
  (qualified_name) @import.module)

; using System;
(using_directive
  (identifier) @import.module)

; using alias = System.Text;
(using_directive
  (name_equals
    (identifier) @import.name)
  (qualified_name) @import.module)

; --- INHERITANCE ---

; class Foo : Bar  (could be base class or interface)
(class_declaration
  name: (identifier) @inherit.child
  (base_list
    (identifier) @inherit.parent))

(class_declaration
  name: (identifier) @inherit.child
  (base_list
    (qualified_name) @inherit.parent))

; interface IFoo : IBar
(interface_declaration
  name: (identifier) @inherit.child
  (base_list
    (identifier) @inherit.parent))

; --- TYPE USES ---

; method return type
(method_declaration
  type: (identifier) @type.use)

; local variable type
(variable_declaration
  type: (identifier) @type.use)
