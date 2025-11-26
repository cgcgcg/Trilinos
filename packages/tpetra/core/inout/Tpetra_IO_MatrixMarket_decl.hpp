#ifndef TPETRA_IO_MATRIXMARKET_DECL_HPP
#define TPETRA_IO_MATRIXMARKET_DECL_HPP

#include "Tpetra_IO_Base_decl.hpp"
#include "Tpetra_IO_Base_def.hpp"
#include "Tpetra_IO_Rank0ReaderWriterBase_decl.hpp"
#include "Tpetra_IO_Rank0ReaderWriterBase_def.hpp"
#include "Tpetra_IO_DistributedReaderWriterBase_decl.hpp"
#include "Tpetra_IO_DistributedReaderWriterBase_def.hpp"

namespace Tpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class MatrixMarketLocalReaderWriter : public LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_multivector_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::multivector_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::vector_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type;

  MatrixMarketLocalReaderWriter(bool tolerant, bool debug);

  FileType detectFileType(const std::string& filename) const;

  Teuchos::RCP<multivector_type> readMultiVector(const std::string& filename) const;

  Teuchos::RCP<map_multivector_type> readMapMultiVector(const std::string& filename) const;

  Teuchos::RCP<crs_matrix_type> readCrsMatrix(const std::string& filename) const;

  void write(const std::string& filename,
             const multivector_type& mv) const;

  void writeMapMultiVector(const std::string& filename,
                           const map_multivector_type& mv) const;

  void write(const std::string& filename,
             const crs_matrix_type& A) const;

 private:
  bool tolerant_;
  bool debug_;
};

template <class Scalar        = ::Tpetra::Details::DefaultTypes::scalar_type,
          class LocalOrdinal  = ::Tpetra::Details::DefaultTypes::local_ordinal_type,
          class GlobalOrdinal = ::Tpetra::Details::DefaultTypes::global_ordinal_type,
          class Node          = ::Tpetra::Details::DefaultTypes::node_type>
class MatrixMarketRank0ReaderWriter : public Rank0ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  MatrixMarketRank0ReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, bool tolerant = false, bool debug = false);
};

template <class Scalar        = ::Tpetra::Details::DefaultTypes::scalar_type,
          class LocalOrdinal  = ::Tpetra::Details::DefaultTypes::local_ordinal_type,
          class GlobalOrdinal = ::Tpetra::Details::DefaultTypes::global_ordinal_type,
          class Node          = ::Tpetra::Details::DefaultTypes::node_type>
class MatrixMarketDistributedReaderWriter : public DistributedReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  MatrixMarketDistributedReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, bool tolerant = false, bool debug = false);
};


}  // namespace Tpetra

#endif
