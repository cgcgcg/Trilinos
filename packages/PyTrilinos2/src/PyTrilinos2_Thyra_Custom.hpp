#ifndef PYTRILINOS2_THYRA_CUSTOM_HPP
#define PYTRILINOS2_THYRA_CUSTOM_HPP

#include <Thyra_LinearOpWithSolveBase_decl.hpp>
// #include <Tpetra_MultiVector.hpp>

template<typename T>
void define_solve(T cl) {
  using SCALAR = typename T::type::scalar_type;
  // using LOCAL_ORDINAL = typename T::type::local_ordinal_type;
  // using GLOBAL_ORDINAL = typename T::type::global_ordinal_type;
  // using NODE = typename T::type::node_type;

  cl.def("solve",[](Teuchos::RCP<Thyra::LinearOpWithSolveBase<SCALAR> > &m,
                    const Thyra::EOpTransp A_trans,
                    const Thyra::MultiVectorBase<SCALAR> &B,
                    const Teuchos::RCP<Thyra::MultiVectorBase<SCALAR> > &X)
    { return m->solve(A_trans, B, X.ptr()); });
  // cl.def("solve",[](Teuchos::RCP<Thyra::LinearOpWithSolveBase<SCALAR> > &m,
  //                   const Thyra::EOpTransp A_trans,
  //                   const Tpetra::MultiVector<SCALAR, LOCAL_ORDINAL, GLOBAL_ORDINAL, NODE> &B,
  //                   const Teuchos::RCP<Tpetra::MultiVector<SCALAR, LOCAL_ORDINAL, GLOBAL_ORDINAL, NODE> > &X)
  //   { return Thyra::solve(m, A_trans, B, X.ptr()); });
}

#endif
