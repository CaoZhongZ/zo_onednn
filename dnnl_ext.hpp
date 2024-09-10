#pragma once

#include <dnnl.h>
#include <dnnl.hpp>

namespace dnnl {
class primitive_ext : public primitive {
public:
  primitive_ext(const primitive base) : primitive(base) {}
  primitive_ext(primitive&& base) : primitive(std::move(base)) {}

  /// Returns a memory descriptor.
  ///
  /// @note
  ///     There are also convenience methods
  ///     #dnnl::primitive_desc_base::src_desc(),
  ///     #dnnl::primitive_desc_base::dst_desc(), and others.
  ///
  /// @param what The kind of parameter to query; can be
  ///     #dnnl::query::src_md, #dnnl::query::dst_md, etc.
  /// @param idx Index of the parameter. For example, convolution bias can
  ///     be queried with what = #dnnl::query::weights_md and idx = 1.
  /// @returns The requested memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     parameter of the specified kind or index.
  const dnnl_memory_desc_t query_md(query what, int idx = 0) const {
      std::vector<query> valid_q {query::src_md, query::diff_src_md,
              query::weights_md, query::diff_weights_md, query::dst_md,
              query::diff_dst_md, query::workspace_md, query::scratchpad_md,
              query::exec_arg_md};
      if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                  [=](query q) { return what == q; }))
          DNNL_THROW_ERROR(dnnl_invalid_arguments,
                  "memory descriptor query is invalid");

      return dnnl_primitive_desc_query_md(
              this->get_primitive_desc(), dnnl::convert_to_c(what), idx);
  }

  /// Returns a source memory descriptor.
  /// @param idx Source index.
  /// @returns Source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     source parameter with index @p idx.
  const dnnl_memory_desc_t src_desc(int idx) const {
      return query_md(query::src_md, idx);
  }

  /// Returns a destination memory descriptor.
  /// @param idx Destination index.
  /// @returns Destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     destination parameter with index @p idx.
  const dnnl_memory_desc_t dst_desc(int idx) const {
      return query_md(query::dst_md, idx);
  }

  /// Returns a weights memory descriptor.
  /// @param idx Weights index.
  /// @returns Weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     weights parameter with index @p idx.
  const dnnl_memory_desc_t weights_desc(int idx) const {
      return query_md(query::weights_md, idx);
  }

  /// Returns a diff source memory descriptor.
  /// @param idx Diff source index.
  /// @returns Diff source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff source parameter with index @p idx.
  const dnnl_memory_desc_t diff_src_desc(int idx) const {
      return query_md(query::diff_src_md, idx);
  }

  /// Returns a diff destination memory descriptor.
  /// @param idx Diff destination index.
  /// @returns Diff destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff destination parameter with index @p idx.
  const dnnl_memory_desc_t diff_dst_desc(int idx) const {
      return query_md(query::diff_dst_md, idx);
  }

  /// Returns a diff weights memory descriptor.
  /// @param idx Diff weights index.
  /// @returns Diff weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff weights parameter with index @p idx.
  const dnnl_memory_desc_t diff_weights_desc(int idx) const {
      return query_md(query::diff_weights_md, idx);
  }

  // Separate versions without the index argument for documentation
  // purposes.

  /// Returns a source memory descriptor.
  /// @returns Source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     source parameter.
  const dnnl_memory_desc_t src_desc() const { return src_desc(0); }

  /// Returns a destination memory descriptor.
  /// @returns Destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     destination parameter.
  const dnnl_memory_desc_t dst_desc() const { return dst_desc(0); }

  /// Returns a weights memory descriptor.
  /// @returns Weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     weights parameter.
  const dnnl_memory_desc_t weights_desc() const { return weights_desc(0); }

  /// Returns a diff source memory descriptor.
  /// @returns Diff source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff source memory with.
  const dnnl_memory_desc_t diff_src_desc() const { return diff_src_desc(0); }

  /// Returns a diff destination memory descriptor.
  /// @returns Diff destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff destination parameter.
  const dnnl_memory_desc_t diff_dst_desc() const { return diff_dst_desc(0); }

  /// Returns a diff weights memory descriptor.
  /// @returns Diff weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff weights parameter.
  const dnnl_memory_desc_t diff_weights_desc() const { return diff_weights_desc(0); }

  /// Returns the workspace memory descriptor.
  /// @returns Workspace memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not require
  ///     workspace parameter.
  const dnnl_memory_desc_t workspace_desc() const {
      return query_md(query::workspace_md, 0);
  }

  /// Returns the scratchpad memory descriptor.
  /// @returns scratchpad memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not require
  ///     scratchpad parameter.
  /// @sa @ref dev_guide_attributes_scratchpad
  const dnnl_memory_desc_t scratchpad_desc() const {
      return query_md(query::scratchpad_md, 0);
  }

  memory make_memory(const dnnl_memory_desc_t memory_desc,
        const engine &aengine, memory_kind kind,
        void *handle = DNNL_MEMORY_ALLOCATE) {
    dnnl_memory_t c_memory;
    error::wrap_c_api(
            dnnl_sycl_interop_memory_create(&c_memory, memory_desc,
                    aengine.get(), convert_to_c(kind), handle),
            "could not create a memory");
    return memory(c_memory);
  }

  memory make_src(const engine &aengine, memory_kind kind, void* handle) {
    return make_memory(src_desc(), aengine, kind, handle);
  }

  memory make_weights(const engine &aengine, memory_kind kind, void* handle) {
    return make_memory(weights_desc(), aengine, kind, handle);
  }

  memory make_dst(const engine &aengine, memory_kind kind, void* handle) {
    return make_memory(dst_desc(), aengine, kind, handle);
  }

  memory make_scratchpad(const engine &aengine, memory_kind kind, void* handle) {
    return make_memory(scratchpad_desc(), aengine, kind, handle);
  }

  memory make_args(int arg_class, const engine& aengine, void *handle) {
    switch(arg_class) {
      case DNNL_ARG_SRC:
        return make_src(aengine, memory_kind::usm, handle);
      case DNNL_ARG_WEIGHTS:
        return make_weights(aengine, memory_kind::usm, handle);
      default:
        throw std::exception();
    }
  }

  void execute(
    const stream& astream,
    const engine& aengine,
    std::vector<std::pair<int, void *>>&& handles
  ) const {
    for (const auto &p : handles) {
      auto it = args.find(p.first);
      if (it == args.end())
        args.emplace(p.first, make_args(p.first, aengine, p.second));
      else
        it->second.set_data_handle(p.second);
    }
    primitive::execute(astream, args);
  }
private:
  std::unordered_map<int, memory> args;
};
}
