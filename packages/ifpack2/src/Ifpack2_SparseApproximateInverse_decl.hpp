// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/// \file Ifpack2_SparseApproximateInverse_decl.hpp
/// \brief Declaration of sparse approximate inverse preconditioner

#ifndef IFPACK2_SPARSEAPPROXIMATEINVERSE_DECL_HPP
#define IFPACK2_SPARSEAPPROXIMATEINVERSE_DECL_HPP

#include "Ifpack2_Preconditioner.hpp"
#include "Ifpack2_Details_CanChangeMatrix.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <type_traits>

namespace Teuchos {
class ParameterList;  // forward declaration
}

namespace Ifpack2 {

/// \class SparseApproximateInverse
/// \brief Sparse approximate inverse of a
///   Tpetra sparse matrix
/// \tparam A specialization of Tpetra::RowMatrix.
///
/// This class computes a sparse approximate inverse of a given sparse matrix
/// represented as a Tpetra::CrsMatrix.
///
/// @remark See the documentation of setParameters() for a list of valid
/// parameters.
///
template <class MatrixType>
class SparseApproximateInverse : virtual public Ifpack2::Preconditioner<typename MatrixType::scalar_type,
                                                                        typename MatrixType::local_ordinal_type,
                                                                        typename MatrixType::global_ordinal_type,
                                                                        typename MatrixType::node_type>,
                                 virtual public Ifpack2::Details::CanChangeMatrix<Tpetra::RowMatrix<typename MatrixType::scalar_type,
                                                                                                    typename MatrixType::local_ordinal_type,
                                                                                                    typename MatrixType::global_ordinal_type,
                                                                                                    typename MatrixType::node_type>> {
 public:
  //! \name Typedefs
  //@{

  //! The type of the entries of the input MatrixType.
  typedef typename MatrixType::scalar_type scalar_type;

  //! The type of local indices in the input MatrixType.
  typedef typename MatrixType::local_ordinal_type local_ordinal_type;

  //! The type of global indices in the input MatrixType.
  typedef typename MatrixType::global_ordinal_type global_ordinal_type;

  //! The Node type used by the input MatrixType.
  typedef typename MatrixType::node_type node_type;

  //! The type of the magnitude (absolute value) of a matrix entry.
  typedef typename Teuchos::ScalarTraits<scalar_type>::magnitudeType magnitude_type;

  //! Type of the Tpetra::RowMatrix specialization that this class uses.
  typedef Tpetra::RowMatrix<scalar_type,
                            local_ordinal_type,
                            global_ordinal_type,
                            node_type>
      row_matrix_type;

  typedef typename row_matrix_type::global_inds_host_view_type global_inds_host_view_type;
  typedef typename row_matrix_type::local_inds_host_view_type local_inds_host_view_type;
  typedef typename row_matrix_type::values_host_view_type values_host_view_type;

  typedef typename row_matrix_type::nonconst_global_inds_host_view_type nonconst_global_inds_host_view_type;
  typedef typename row_matrix_type::nonconst_local_inds_host_view_type nonconst_local_inds_host_view_type;
  typedef typename row_matrix_type::nonconst_values_host_view_type nonconst_values_host_view_type;

  static_assert(std::is_same<MatrixType, row_matrix_type>::value, "Ifpack2::SparseApproximateInverse: The template parameter MatrixType must be a Tpetra::RowMatrix specialization.  Please don't use Tpetra::CrsMatrix (a subclass of Tpetra::RowMatrix) here anymore.  The constructor can take either a RowMatrix or a CrsMatrix just fine.");

  //! Type of the Tpetra::CrsMatrix specialization that this class uses.
  typedef Tpetra::CrsMatrix<scalar_type,
                            local_ordinal_type,
                            global_ordinal_type,
                            node_type>
      crs_matrix_type;

  //! Type of the Tpetra::Map specialization that this class uses.
  typedef Tpetra::Map<local_ordinal_type,
                      global_ordinal_type,
                      node_type>
      map_type;
  //@}
  //! \name Constructors and Destructors
  //@{

  /// \brief Constructor
  ///
  /// \param A [in] The sparse matrix to factor, as a
  ///   Tpetra::RowMatrix.  (Tpetra::CrsMatrix inherits from this, so
  ///   you may use a Tpetra::CrsMatrix here instead.)
  ///
  /// The factorization will <i>not</i> modify the input matrix.  It
  /// stores approximate inverse separately.
  explicit SparseApproximateInverse(const Teuchos::RCP<const row_matrix_type>& A);

  //! Destructor
  virtual ~SparseApproximateInverse() = default;

  //@}
  //! \name Methods for setting up and computing the approximate inverse
  //@{

  /// \brief Set preconditioner parameters.
  ///
  /// SparseApproximateInverse implements the following parameters:
  /// <ul>
  /// <li> "fact: drop tolerance" (\c magnitude_type)
  /// </ul>
  /// "fact: drop tolerance" is the magnitude threshold for dropping
  /// entries.
  void setParameters(const Teuchos::ParameterList& params);

  /// \brief Clear any previously computed inverse.
  ///
  /// You may call this before calling compute().  The compute()
  /// method will call this automatically if it has not yet been
  /// called.  If you call this after calling compute(), you must
  /// recompute the inversion (by calling compute() again) before
  /// you may call apply().
  void initialize();

  //! Returns \c true if the preconditioner has been successfully initialized.
  inline bool isInitialized() const {
    return IsInitialized_;
  }

  //! Compute the sparse approximate inverse.
  void compute();

  //! If compute() is completed, this query returns true, otherwise it returns false.
  inline bool isComputed() const {
    return IsComputed_;
  }

  //@}
  //! \name Implementation of Ifpack2::Details::CanChangeMatrix
  //@{

  /// \brief Change the matrix to be preconditioned.
  ///
  /// \param A [in] The new matrix.
  ///
  /// \post <tt>! isInitialized ()</tt>
  /// \post <tt>! isComputed ()</tt>
  ///
  /// Calling this method resets the preconditioner's state.  After
  /// calling this method with a nonnull input, you must first call
  /// initialize() and compute() (in that order) before you may call
  /// apply().
  ///
  /// You may call this method with a null input.  If A is null, then
  /// you may not call initialize() or compute() until you first call
  /// this method again with a nonnull input.  This method invalidates
  /// any previous factorization whether or not A is null, so calling
  /// setMatrix() with a null input is one way to clear the
  /// preconditioner's state (and free any memory that it may be
  /// using).
  ///
  /// The new matrix A need not necessarily have the same Maps or even
  /// the same communicator as the original matrix.
  virtual void
  setMatrix(const Teuchos::RCP<const row_matrix_type>& A);

  //@}
  //! \name Implementation of Tpetra::Operator
  //@{

  /// \brief Apply the sparse approximate inverse preconditioner to X, resulting in Y.
  ///
  /// \param X [in] Input multivector; "right-hand side" of the solve.
  /// \param Y [out] Output multivector; result of the solve.
  /// \param mode [in] Whether to apply the transpose (Teuchos::NO_TRANS, Teuchos::TRANS, Teuchos::CONJ_TRANS).
  /// \param alpha [in] Scaling factor for the result.
  /// \param beta [in] Scaling factor for Y before adding the result.
  void
  apply(const Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& X,
        Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha     = Teuchos::ScalarTraits<scalar_type>::one(),
        scalar_type beta      = Teuchos::ScalarTraits<scalar_type>::zero()) const;

  //! Tpetra::Map representing the domain of this operator.
  Teuchos::RCP<const map_type> getDomainMap() const;

  //! Tpetra::Map representing the range of this operator.
  Teuchos::RCP<const map_type> getRangeMap() const;

  //! Whether this object's apply() method can apply the transpose (or conjugate transpose, if applicable).
  bool hasTransposeApply() const;

  //@}
  //! \name Mathematical functions
  //@{

  //! Returns the input matrix's communicator.
  Teuchos::RCP<const Teuchos::Comm<int>> getComm() const;

  //! Returns a reference to the matrix to be preconditioned.
  Teuchos::RCP<const row_matrix_type> getMatrix() const;

  //! Returns the number of calls to Initialize().
  int getNumInitialize() const;

  //! Returns the number of calls to Compute().
  int getNumCompute() const;

  //! Returns the number of calls to apply().
  int getNumApply() const;

  //! Returns the time spent in Initialize().
  double getInitializeTime() const;

  //! Returns the time spent in Compute().
  double getComputeTime() const;

  //! Returns the time spent in apply().
  double getApplyTime() const;

  //! Get a rough estimate of cost per iteration
  size_t getNodeSmootherComplexity() const;

  //! Gets the dropping tolerance
  inline magnitude_type getDropTolerance() const {
    return (DropTolerance_);
  }

  //! Returns the number of nonzero entries in the global graph.
  global_size_t getGlobalNumEntries() const;

  //! Returns the number of nonzero entries in the local graph.
  size_t getLocalNumEntries() const;

  //@}
  //! \name Implementation of Teuchos::Describable
  //@{

  /** \brief Return a simple one-line description of this object. */
  std::string description() const;

  /** \brief Print the object with some verbosity level to an FancyOStream object. */
  void describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const;

  //@}

 private:
  typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> MV;
  typedef Teuchos::ScalarTraits<scalar_type> STS;
  typedef Teuchos::ScalarTraits<magnitude_type> STM;
  typedef typename Teuchos::Array<local_ordinal_type>::size_type size_type;

  //! Copy constructor (declared private and undefined; may not be used)
  SparseApproximateInverse(const SparseApproximateInverse<MatrixType>& RHS);

  //! operator= (declared private and undefined; may not be used)
  SparseApproximateInverse<MatrixType>& operator=(const SparseApproximateInverse<MatrixType>& RHS);

  // \name The matrix and its incomplete LU factors
  //@{

  //! The matrix to be preconditioned.
  Teuchos::RCP<const row_matrix_type> A_;
  //! The sparse approximate inverse of A.
  Teuchos::RCP<const row_matrix_type> invA_;

  //@}
  // \name Parameters (set by setParameters())
  //@{

  //! Discard all elements below this tolerance
  magnitude_type DropTolerance_;

  //@}
  // \name Other internal data
  //@{

  //! Total time in seconds for all successful calls to initialize().
  double InitializeTime_;
  //! Total time in seconds for all successful calls to compute().
  double ComputeTime_;
  //! Total time in seconds for all successful calls to apply().
  mutable double ApplyTime_;
  //! The number of successful calls to initialize().
  int NumInitialize_;
  //! The number of successful call to compute().
  int NumCompute_;
  //! The number of successful call to apply().
  mutable int NumApply_;
  //! \c true if \c this object has been initialized
  bool IsInitialized_;
  //! \c true if \c this object has been computed
  bool IsComputed_;
  //@}

};  // class SparseApproximateInverse

//! Computes the sparse approximate inverse with graph sparsityPattern of the matrix A
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
GetSparseApproximateInverse(const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A,
                            const Teuchos::RCP<const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>>& sparsityPattern);

}  // namespace Ifpack2

#endif /* IFPACK2_SPARSEAPPROXIMATEINVERSE_DECL_HPP */
