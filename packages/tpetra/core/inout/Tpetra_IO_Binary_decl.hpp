#ifndef TPETRA_IO_BINARY_DECL_HPP
#define TPETRA_IO_BINARY_DECL_HPP

#include "Tpetra_IO_Base_decl.hpp"
#include "Tpetra_IO_Base_def.hpp"
#include "Tpetra_IO_Rank0ReaderWriterBase_decl.hpp"
#include "Tpetra_IO_Rank0ReaderWriterBase_def.hpp"
#include "Tpetra_IO_DistributedReaderWriterBase_decl.hpp"
#include "Tpetra_IO_DistributedReaderWriterBase_def.hpp"

namespace Tpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal>
struct BinaryDataTraits {
  using scalar_type = double;
  using index_type  = int;
};

template <class LocalOrdinal, class GlobalOrdinal>
struct BinaryDataTraits<GlobalOrdinal, LocalOrdinal, GlobalOrdinal> {
  using scalar_type = int;
  using index_type  = int;
};

template <class LocalOrdinal, class GlobalOrdinal>
struct BinaryDataTraits<LocalOrdinal, LocalOrdinal, GlobalOrdinal> {
  using scalar_type = int;
  using index_type  = int;
};

template <class LocalOrdinal, class GlobalOrdinal>
struct BinaryDataTraits<std::complex<double>, LocalOrdinal, GlobalOrdinal> {
  using scalar_type = std::complex<double>;
  using index_type  = int;
};

template <class LocalOrdinal, class GlobalOrdinal>
struct BinaryDataTraits<std::complex<float>, LocalOrdinal, GlobalOrdinal> {
  using scalar_type = std::complex<double>;
  using index_type  = int;
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class BinaryLocalReaderWriter : public LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_multivector_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::multivector_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::vector_type;
  using typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type;

  BinaryLocalReaderWriter();

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
};

template <class Scalar        = ::Tpetra::Details::DefaultTypes::scalar_type,
          class LocalOrdinal  = ::Tpetra::Details::DefaultTypes::local_ordinal_type,
          class GlobalOrdinal = ::Tpetra::Details::DefaultTypes::global_ordinal_type,
          class Node          = ::Tpetra::Details::DefaultTypes::node_type>
class BinaryRank0ReaderWriter : public Rank0ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  BinaryRank0ReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm);
};

template <class Scalar        = ::Tpetra::Details::DefaultTypes::scalar_type,
          class LocalOrdinal  = ::Tpetra::Details::DefaultTypes::local_ordinal_type,
          class GlobalOrdinal = ::Tpetra::Details::DefaultTypes::global_ordinal_type,
          class Node          = ::Tpetra::Details::DefaultTypes::node_type>
class BinaryDistributedReaderWriter : public DistributedReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  BinaryDistributedReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm);
};


}  // namespace Tpetra

#endif
