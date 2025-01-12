#include <dnnl_ext.hpp>

enum joint_dtypes {
  _fp32 = 0,
  _f16, _bf16, int8, _f16_int4, _bf16_int4
};

enum transposes {
  _nn = 0, _nt, _tn, _tt
};

template <enum transposes tr, enum joint_dtypes>
struct onednn_tag_mapper;

template <enum joint_dtypes Ts>
struct onednn_tag_mapper<_nt> {
  static inline std::tuple<dnnl::tag, dnnl::tag> do() {
    return make_tuple(memory::format_tag::ab, memory::format_tag::ba);
  };
};

template <enum joint_dtypes Ts>
struct onednn_tag_mapper<_nn> {
  static_assert(Ts != _f16_int4 || Ts != _bf16_int4, "Reason why");
  static inline std::tuple<dnnl::tag, dnnl::tag> do() {
    return make_tuple(memory::format_tag::ab, memory::format_tag::ba);
  };
};

template <enum joint_dtypes Ts>
struct onednn_types_mapper;

template <> struct onednn_types_mapper<_f16_int4> {
  static inline std::tuple<
    memory::data_type,
    memory::data_type,
    memory::data_type,
    memory::data_type> do() {
      return make_tuple(memory::data_type::f16,
          memory::data_type::u4, memory::data_type::f16, memory::data_type::f16);
    }
};

template <> struct onednn_types_mapper<_bf16_int4> {
  static inline std::tuple<
    memory::data_type,
    memory::data_type,
    memory::data_type,
    memory::data_type> do() {
      return make_tuple(memory::data_type::bf16,
          memory::data_type::u4, memory::data_type::bf16, memory::data_type::bf16);
    }
};

template <transposes tr, enum joint_dtypes, bool bias, F f_attr>
struct create_matmul {
  static primitive_ext run(int device_id) {
    // write everything inside this function with if constexpr clauses
  }
private:
  // if default constructor of primitive cache could read the environment variable
  // then it'll save a lot of trouble
  static thread_local std::array<primitive_cache, 16> mappings {};

  // this won't be needed if primitive_cache have good default constructor
  static primitive_cache& get_cache(int device_id) {
    auto mapping = mappings[device_id];
    if (!mapping.initialized()) {
      // construct the cache
    } else
      return mappings[device_id];
  }
}

template <typename F>
primitive_ext create_matmul(joint_data_types types, transposes tr, int device_id, F f_attr) {
  switch(types) {
  case _f16_int4:
    return create_matmul<_f16_int4, F>(tr, device_id, f_attr);
  case _bf16_int4:
    return create_matmul<_bf16_int4, F>(tr, device_id, f_attr);
  default:
    throw std::exception();
  }
}

template <enum joint_data_types types, int device_id, typename F>
primitve_ext create_matmul(transposes tr, F f_attr) {
  switch(tr) {
  case _nn:
    return create_matmul<types, device_id, _nn, F>(f_attr);
  case _nt:
    return create_matmul<types, device_id, _nt, F>(f_attr);
  case _tn:
    return create_matmul<types, device_id, _tn, F>(f_attr);
  case _tt:
    return create_matmul<types, device_id, _tt, F>(f_attr);
  default:
    throw std::exception();
  }
}

template <enum joint_data_types types, int device_id, transposes tr, typename F>
primitve_ext create_matmul(F f_attr) {
}

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
