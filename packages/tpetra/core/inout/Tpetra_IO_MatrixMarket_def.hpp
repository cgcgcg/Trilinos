#ifndef TPETRA_IO_MATRIXMARKET_DEF_HPP
#define TPETRA_IO_MATRIXMARKET_DEF_HPP

#include "Tpetra_IO_MatrixMarket_decl.hpp"
#include "Teuchos_MatrixMarket_Raw_Reader.hpp"
#include "Teuchos_MatrixMarket_Raw_Writer.hpp"

namespace Tpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MatrixMarketLocalReaderWriter(const bool tolerant, const bool debug)
  : tolerant_(tolerant)
  , debug_(debug) {
  // nothing
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
FileType
MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    detectFileType(const std::string& filename) const {
  std::ifstream file;
  std::string line0;
  std::string line1;
  file.open(filename);
  if (file.is_open()) {
    getline(file, line0);
    getline(file, line1);
    if (line0 == "%%MatrixMarket matrix array real general")
      return MultiVectorFile;
    else if (line0 == "%%MatrixMarket matrix array complex general")
      return MultiVectorFile;
    else if (line0 == "%%MatrixMarket matrix array integer general")
      return MapFile;
    else if (line0 == "%%MatrixMarket matrix coordinate real general")
      return CrsMatrixFile;
    else if (line0 == "%%MatrixMarket matrix coordinate complex general")
      return CrsMatrixFile;
    else
      return UnknownFile;
  } else
    return UnopenableFile;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::multivector_type>
MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readMultiVector(const std::string& filename) const {
  std::ifstream file;
  std::string line;
  file.open(filename);
  if (file.is_open()) {
    getline(file, line);
    getline(file, line);
    auto items   = Teuchos::StrUtils::stringTokenizer(line);
    auto numRows = Teuchos::StrUtils::atoi(items[0]);
    auto numCols = Teuchos::StrUtils::atoi(items[1]);
    auto map     = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numRows);
    auto mv      = Teuchos::rcp(new multivector_type(map, numCols));
    auto lcl     = mv->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (LocalOrdinal col = 0; col < numCols; ++col) {
      for (LocalOrdinal row = 0; row < numRows; ++row) {
        getline(file, line);
        lcl(row, col) = Teuchos::StrUtils::atoi(line);
      }
    }
    file.close();
    return mv;
  } else
    return Teuchos::null;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_multivector_type>
MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readMapMultiVector(const std::string& filename) const {
  std::ifstream file;
  std::string line;
  file.open(filename);
  if (file.is_open()) {
    getline(file, line);
    getline(file, line);
    auto items   = Teuchos::StrUtils::stringTokenizer(line);
    auto numRows = Teuchos::StrUtils::atoi(items[0]);
    auto numCols = Teuchos::StrUtils::atoi(items[1]);
    TEUCHOS_ASSERT(numCols == 2);
    auto map = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numRows);
    auto mv  = Teuchos::rcp(new map_multivector_type(map, numCols));
    auto lcl = mv->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (LocalOrdinal row = 0; row < numRows; ++row) {
      for (LocalOrdinal col = 0; col < numCols; ++col) {
        getline(file, line);
        lcl(row, col) = Teuchos::StrUtils::atoi(line);
      }
    }
    file.close();
    return mv;
  } else
    return Teuchos::null;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type>
MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readCrsMatrix(const std::string& filename) const {
  bool isComplex = false;
  {
    std::ifstream file;
    file.open(filename);
    if (file.is_open()) {
      std::string line;
      getline(file, line);
      file.close();
      isComplex = (line == "%%MatrixMarket matrix coordinate complex general");
    }
  }
  LocalOrdinal numRows = 0;
  LocalOrdinal numCols = 0;
  Teuchos::ArrayRCP<LocalOrdinal> rowptr_int;
  Teuchos::ArrayRCP<size_t> rowptr;
  Teuchos::ArrayRCP<LocalOrdinal> colind;
  Teuchos::ArrayRCP<Scalar> values;
  if (!isComplex || Teuchos::ScalarTraits<Scalar>::isComplex) {
    Teuchos::MatrixMarket::Raw::Reader<Scalar, LocalOrdinal> mm_reader(tolerant_, debug_);
    mm_reader.readFile(rowptr_int, colind, values, numRows, numCols, filename);
  } else {
    Teuchos::MatrixMarket::Raw::Reader<Scalar, LocalOrdinal> mm_reader(tolerant_, debug_);
    mm_reader.readFile(rowptr_int, colind, values, numRows, numCols, filename);
  }
  rowptr.resize(rowptr_int.size());
  std::copy(rowptr_int.begin(), rowptr_int.end(), rowptr.begin());

  auto rangemap = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numRows);
  Teuchos::RCP<const map_type> domainmap;
  if (numRows != numCols)
    domainmap = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numCols);
  else
    domainmap = rangemap;

  auto crs = Teuchos::rcp(new crs_matrix_type(rangemap, domainmap, rowptr, colind, values));
  crs->fillComplete(domainmap, rangemap);
  return crs;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const multivector_type& mv) const {
  auto lcl             = mv.getLocalViewHost(Tpetra::Access::ReadOnly);
  LocalOrdinal numRows = lcl.extent(0);
  LocalOrdinal numCols = lcl.extent(1);
  std::ofstream file;
  file.open(filename);
  file << "%%MatrixMarket matrix array real general\n";
  file << numRows << " " << numCols << std::endl;
  for (LocalOrdinal col = 0; col < numCols; ++col) {
    for (LocalOrdinal row = 0; row < numRows; ++row) {
      file << lcl(row, col) << std::endl;
    }
  }
  file.close();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    writeMapMultiVector(const std::string& filename,
                        const map_multivector_type& mv) const {
  auto lcl             = mv.getLocalViewHost(Tpetra::Access::ReadOnly);
  LocalOrdinal numRows = lcl.extent(0);
  LocalOrdinal numCols = lcl.extent(1);
  std::ofstream file;
  file.open(filename);
  file << "%%MatrixMarket matrix array integer general\n";
  file << numRows << " " << numCols << std::endl;
  for (LocalOrdinal row = 0; row < numRows; ++row) {
    for (LocalOrdinal col = 0; col < numCols; ++col) {
      file << lcl(row, col) << std::endl;
    }
  }
  file.close();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const crs_matrix_type& A) const {
  LocalOrdinal numRows = A.getGlobalNumRows();
  LocalOrdinal numCols = A.getGlobalNumCols();
  auto lclA            = A.getLocalMatrixHost();
  auto k_rowptr        = Kokkos::View<LocalOrdinal*, Kokkos::HostSpace>("", lclA.numRows() + 1);
  Kokkos::deep_copy(k_rowptr, lclA.graph.row_map);
  auto rowptr = Kokkos::Compat::getArrayView(k_rowptr);
  auto colind = Kokkos::Compat::getArrayView(lclA.graph.entries);
  auto values = Kokkos::Compat::getArrayView(lclA.values);
  Teuchos::MatrixMarket::Raw::Writer<Scalar, LocalOrdinal> mm_writer;
  mm_writer.writeFile(filename, rowptr(), colind(), values(), numRows, numCols);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
MatrixMarketRank0ReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    MatrixMarketRank0ReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, const bool tolerant, const bool debug)
  : Rank0ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm, Teuchos::rcp(new MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tolerant, debug))) {
  // nothing
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
MatrixMarketDistributedReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    MatrixMarketDistributedReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, const bool tolerant, const bool debug)
  : DistributedReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm, Teuchos::rcp(new MatrixMarketLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tolerant, debug))) {
  // nothing
}
}  // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#define TPETRA_IO_MATRIXMARKET_INSTANT(SCALAR, LO, GO, NODE)          \
  template class MatrixMarketLocalReaderWriter<SCALAR, LO, GO, NODE>; \
  template class MatrixMarketRank0ReaderWriter<SCALAR, LO, GO, NODE>; \
  template class MatrixMarketDistributedReaderWriter<SCALAR, LO, GO, NODE>;

#endif
