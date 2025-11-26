#ifndef TPETRA_IO_BASE_DECL_HPP
#define TPETRA_IO_BASE_DECL_HPP

#include "Teuchos_Assert.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Tpetra_Access.hpp"
#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include <stdexcept>
#include <string>
#include <any>
#include "Teuchos_TestForException.hpp"
#include "Tpetra_CombineMode.hpp"

#include "mpi.h"

namespace Tpetra {

enum ReadWriteMaps : int {
  None,
  Auto,
  All
};

enum FileType : int {
  UnknownFile = 999,
  UnopenableFile = 666,
  MapFile = 1000,
  VectorFile = 1001,
  MultiVectorFile = 1002,
  CrsMatrixFile = 1003
};


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class LocalReaderWriterBase {
 public:
  using map_type             = Map<LocalOrdinal, GlobalOrdinal, Node>;
  using map_multivector_type = MultiVector<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node>;
  using multivector_type     = MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using vector_type          = Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using crs_matrix_type      = CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  virtual Teuchos::RCP<multivector_type> readMultiVector(const std::string& filename) const = 0;

  virtual Teuchos::RCP<map_multivector_type> readMapMultiVector(const std::string& filename) const = 0;

  Teuchos::RCP<vector_type> readVector(const std::string& filename) const;

  virtual Teuchos::RCP<crs_matrix_type> readCrsMatrix(const std::string& filename) const = 0;

  virtual FileType detectFileType(const std::string& filename) const = 0;

  virtual void write(const std::string& filename,
                     const multivector_type& mv) const = 0;

  void write(const std::string& filename,
             const vector_type& v) const;

  virtual void writeMapMultiVector(const std::string& filename,
                                   const map_multivector_type& mv) const = 0;

  virtual void write(const std::string& filename,
                     const crs_matrix_type& A) const = 0;

  Teuchos::RCP<const map_type> buildLocalMap(LocalOrdinal numUnknowns) const;
};

template <class Scalar        = ::Tpetra::Details::DefaultTypes::scalar_type,
          class LocalOrdinal  = ::Tpetra::Details::DefaultTypes::local_ordinal_type,
          class GlobalOrdinal = ::Tpetra::Details::DefaultTypes::global_ordinal_type,
          class Node          = ::Tpetra::Details::DefaultTypes::node_type>
class ReaderWriterBase {
 public:
  using map_type             = Map<LocalOrdinal, GlobalOrdinal, Node>;
  using map_multivector_type = MultiVector<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node>;
  using multivector_type     = MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using vector_type          = Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using crs_matrix_type      = CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using import_type          = Import<LocalOrdinal, GlobalOrdinal, Node>;

  ReaderWriterBase(const Teuchos::RCP<const Teuchos::Comm<int>>& comm);

  Teuchos::RCP<const Teuchos::Comm<int>> getComm() const;

  virtual Teuchos::RCP<const map_type> readMap(const std::string& filename) const = 0;

  virtual Teuchos::RCP<multivector_type> readMultiVector(const std::string& filename,
                                                         Teuchos::RCP<const map_type> map) const = 0;

  Teuchos::RCP<multivector_type> readMultiVector(const std::string& filename,
                                                 ReadWriteMaps readMapFile) const;

  Teuchos::RCP<multivector_type> readMultiVector(const std::string& filename,
                                                 std::string readMapFile) const;

  virtual Teuchos::RCP<crs_matrix_type> readCrsMatrix(const std::string& filename,
                                                      Teuchos::RCP<const map_type> rowMap    = Teuchos::null,
                                                      Teuchos::RCP<const map_type> colMap    = Teuchos::null,
                                                      Teuchos::RCP<const map_type> domainMap = Teuchos::null,
                                                      Teuchos::RCP<const map_type> rangeMap  = Teuchos::null) const = 0;

  Teuchos::RCP<crs_matrix_type> readCrsMatrix(const std::string& filename,
                                              ReadWriteMaps readMapFiles) const;

  Teuchos::RCP<crs_matrix_type> readCrsMatrix(const std::string& filename,
                                              std::string readMapFiles) const;

  virtual FileType detectFileType(const std::string& filename) const = 0;

  std::any read(const std::string& filename,
                              ReadWriteMaps readMapFiles) const;

  std::any read(const std::string& filename,
                              std::string readMapFiles) const;

  virtual void write(const std::string& filename,
                     const map_type& map) const = 0;

  virtual void write(const std::string& filename,
                     const multivector_type& mv) const = 0;

  void write(const std::string& filename,
             const multivector_type& mv,
             ReadWriteMaps writeMap) const;

  void write(const std::string& filename,
             const multivector_type& mv,
             std::string writeMap) const;

  virtual void write(const std::string& filename,
                     const crs_matrix_type& crs) const = 0;

  void write(const std::string& filename,
             const crs_matrix_type& crs,
             ReadWriteMaps writeMaps) const;

  void write(const std::string& filename,
             const crs_matrix_type& crs,
             std::string writeMaps) const;

  void write(const std::string& filename,
             const std::any& obj,
             ReadWriteMaps writeMaps) const;

  void write(const std::string& filename,
             const std::any& obj,
             std::string writeMaps) const;

  Teuchos::RCP<map_multivector_type> mapToMultiVector(const map_type& map) const;

 private:
  Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

}  // namespace Tpetra

#endif
