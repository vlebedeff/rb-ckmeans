#include "ruby.h"

VALUE rb_return_nil(VALUE self);

void Init_extensions(void) {
    VALUE ckmeans_module = rb_const_get(rb_cObject, rb_intern("Ckmeans"));

    rb_define_singleton_method(ckmeans_module, "c_do_nothing", rb_return_nil, 0);
}

VALUE rb_return_nil(VALUE self) {
    return Qnil;
}
