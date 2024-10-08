#include <dnnl_ext.hpp>

template <typename F>
primitive_ext create_matmul_int4(const tensor... tensors, F f_attr) {
  auto key = shapes_of_all_tensors(tensors);
  static thread_local primitive_cache cache(1024);

  if (cache.find(key) == cache.end()) {
    //
    // slow region, no performance restrictions
    //
    primitive_attr attr;
    f_attr(attr);

    auto matmul_int4 = [[onednn create sequence ...]] onednn_matmul(attr);

    primitive_ext matmul_int4_ext(matmul_int4);
    cache.insert(key, primitive_int4_ext);
    //
    // end no performance restrictions
    //
    return matmul_int4_ext;
  } else
    return key->second;
}

// Users could just do a simple wrapper for the exposed symbols to python
void matmul_int4(const tensor ...) {
  auto executable = create_matmul_int4(tensors...,[]{});

  // Lack of query facility requires following interface
  executable.set_args(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale.data_ptr(),
      [&]() {
        return get_onednn_md(scale);
      });
  executable.set_args(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp.data_ptr(),
      [&]() {
        return get_onednn_md(zp);
      });
  executable(stream, engine,
   {{DNNL_ARG_SRC, src.data_ptr()},{DNNL_ARG_WEIGHT, weight.data_ptr()}, {..} });
}

void matmul_int4_silu(const tensor...) {
  auto silu = [&](primitive_attr& attr) {
    post_op op;
    op.append_elt...

    attr.set_post_ops(op);
  };
  auto executable = create_matmul_int4(tensors..., silu);
  executable.set_args(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale.data_ptr(),
      [&]() {
        return get_onednn_md(scale);
      });
  executable.set_args(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp.data_ptr(),
      [&]() {
        return get_onednn_md(zp);
      });
  executable(stream, engine,
   {{DNNL_ARG_SRC, src.data_ptr()}, {DNNL_ARG_WEIGHT, weight.data_ptr()} });
}

void matmul_int4_resadd(const tensor ...) {
  auto resadd = [&](primitive_attr& attr) {
    post_op op;
    op.append_elt...

    attr.set_post_ops(op);
  };
  auto executable = create_matmul_int4(tensors..., resadd);
  executable.set_args(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale.data_ptr(),
      [&]() {
        return get_onednn_md(scale);
      });
  executable.set_args(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp.data_ptr(),
      [&]() {
        return get_onednn_md(zp);
      });
  executable(stream, engine,
   {{DNNL_ARG_SRC, src.data_ptr()}, {DNNL_ARG_WEIGHT, weight.data_ptr()} });
}
