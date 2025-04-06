#include "ruby.h"

VALUE rb_return_nil() {
    return Qnil;
}

void Init_extensions() {
    VALUE ckmeans_module = rb_const_get(rb_cObject, rb_intern("Ckmeans"));

    rb_define_singleton_method(ckmeans_module, "c_do_nothing", rb_return_nil, 0);
}
