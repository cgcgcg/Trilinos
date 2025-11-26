#ifndef TPETRA_IO_BINARY_DEF_HPP
#define TPETRA_IO_BINARY_DEF_HPP

#include "Tpetra_IO_Binary_decl.hpp"

namespace Tpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BinaryLocalReaderWriter() = default;

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
FileType
BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    detectFileType(const std::string& filename) const {
  std::ifstream file;
  file.open(filename, std::ios::binary);
  if (file.is_open()) {
    FileType fileType;
    file.read(reinterpret_cast<char*>(&fileType), sizeof(FileType));
    if ((fileType == MapFile) || (fileType == MultiVectorFile) || (fileType == CrsMatrixFile))
      return fileType;
    else
      return UnknownFile;
  } else
    return UnopenableFile;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::multivector_type>
BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readMultiVector(const std::string& filename) const {
  using scalar_type = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::scalar_type;
  using index_type  = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::index_type;

  std::ifstream ifs(filename, std::ios::binary);
  TEUCHOS_ASSERT(ifs.good());

  FileType fileType;
  index_type numRows;
  index_type numVectors;
  ifs.read(reinterpret_cast<char*>(&fileType), sizeof(FileType));
  TEUCHOS_ASSERT(fileType == MultiVectorFile);
  ifs.read(reinterpret_cast<char*>(&numRows), sizeof(index_type));
  ifs.read(reinterpret_cast<char*>(&numVectors), sizeof(index_type));

  Kokkos::View<scalar_type**, Kokkos::HostSpace> values_file("values_file", numRows, numVectors);

  ifs.read(reinterpret_cast<char*>(values_file.data()), numRows * numVectors * sizeof(scalar_type));

  typename multivector_type::dual_view_type::t_dev values("values", numRows, numVectors);

  Kokkos::deep_copy(values, values_file);

  auto map = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numRows);
  auto mv  = Teuchos::rcp(new multivector_type(map, values));
  return mv;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_multivector_type>
BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readMapMultiVector(const std::string& filename) const {
  using index_type = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::index_type;

  std::ifstream ifs(filename, std::ios::binary);
  TEUCHOS_ASSERT(ifs.good());

  FileType fileType;
  index_type numRows;
  index_type numVectors;
  ifs.read(reinterpret_cast<char*>(&fileType), sizeof(FileType));
  TEUCHOS_ASSERT(fileType == MapFile);
  ifs.read(reinterpret_cast<char*>(&numRows), sizeof(index_type));
  ifs.read(reinterpret_cast<char*>(&numVectors), sizeof(index_type));

  Kokkos::View<index_type**, Kokkos::HostSpace> values_file("values_file", numRows, numVectors);

  ifs.read(reinterpret_cast<char*>(values_file.data()), numRows * numVectors * sizeof(index_type));

  typename map_multivector_type::dual_view_type::t_dev values("values_double", numRows, numVectors);

  Kokkos::deep_copy(values, values_file);

  auto map = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numRows);
  auto mv  = Teuchos::rcp(new map_multivector_type(map, values));
  return mv;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type>
BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readCrsMatrix(const std::string& filename) const {
  using scalar_type       = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::scalar_type;
  using index_type        = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::index_type;
  using local_graph_type  = typename crs_matrix_type::local_graph_device_type;
  using local_matrix_type = typename crs_matrix_type::local_matrix_device_type;

  std::ifstream ifs(filename, std::ios::binary);
  TEUCHOS_ASSERT(ifs.good());

  FileType fileType;
  index_type numRows;
  index_type numCols;
  index_type nnz;
  ifs.read(reinterpret_cast<char*>(&fileType), sizeof(FileType));
  TEUCHOS_ASSERT(fileType == CrsMatrixFile);
  ifs.read(reinterpret_cast<char*>(&numRows), sizeof(index_type));
  ifs.read(reinterpret_cast<char*>(&numCols), sizeof(index_type));
  ifs.read(reinterpret_cast<char*>(&nnz), sizeof(index_type));

  Kokkos::View<index_type*, Kokkos::HostSpace> rowptr_file("rowptr_file", numRows + 1);
  Kokkos::View<index_type*, Kokkos::HostSpace> colidx_file("colidx_file", nnz);
  Kokkos::View<scalar_type*, Kokkos::HostSpace> values_file("values_file", nnz);

  ifs.read(reinterpret_cast<char*>(rowptr_file.data()), (numRows + 1) * sizeof(index_type));
  ifs.read(reinterpret_cast<char*>(colidx_file.data()), nnz * sizeof(index_type));
  ifs.read(reinterpret_cast<char*>(values_file.data()), nnz * sizeof(scalar_type));

  typename local_graph_type::row_map_type::non_const_type rowptr("rowptr_int", numRows + 1);
  typename local_graph_type::entries_type::non_const_type colidx("colidx_int", nnz);
  typename local_matrix_type::values_type::non_const_type values("values_double", nnz);

  Kokkos::deep_copy(rowptr, rowptr_file);
  Kokkos::deep_copy(colidx, colidx_file);
  Kokkos::deep_copy(values, values_file);

  auto rangemap = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numRows);
  Teuchos::RCP<const map_type> domainmap;
  if (numRows != numCols)
    domainmap = LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(numCols);
  else
    domainmap = rangemap;

  return Teuchos::rcp(new crs_matrix_type(rangemap, domainmap, rowptr, colidx, values));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const multivector_type& mv) const {
  using scalar_type = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::scalar_type;
  using index_type  = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::index_type;

  auto lcl           = mv.getLocalViewHost(Tpetra::Access::ReadOnly);
  index_type numRows = lcl.extent(0);
  index_type numCols = lcl.extent(1);
  Kokkos::View<scalar_type**, Kokkos::HostSpace> values_file("values_file", numRows, numCols);
  Kokkos::deep_copy(values_file, lcl);
  std::ofstream file(filename, std::ios::binary);
  FileType fileType = MultiVectorFile;
  file.write(reinterpret_cast<char*>(&fileType), sizeof(FileType));
  file.write(reinterpret_cast<char*>(&numRows), sizeof(index_type));
  file.write(reinterpret_cast<char*>(&numCols), sizeof(index_type));
  file.write(reinterpret_cast<char*>(values_file.data()), numRows * numCols * sizeof(scalar_type));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    writeMapMultiVector(const std::string& filename,
                        const map_multivector_type& mv) const {
  using index_type = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::index_type;

  auto lcl           = mv.getLocalViewHost(Tpetra::Access::ReadOnly);
  index_type numRows = lcl.extent(0);
  index_type numCols = lcl.extent(1);
  Kokkos::View<index_type**, Kokkos::HostSpace> values_file("values_file", numRows, numCols);
  Kokkos::deep_copy(values_file, lcl);
  std::ofstream file(filename, std::ios::binary);
  FileType fileType = MapFile;
  file.write(reinterpret_cast<char*>(&fileType), sizeof(FileType));
  file.write(reinterpret_cast<char*>(&numRows), sizeof(index_type));
  file.write(reinterpret_cast<char*>(&numCols), sizeof(index_type));
  file.write(reinterpret_cast<char*>(values_file.data()), numRows * numCols * sizeof(index_type));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const crs_matrix_type& A) const {
  using scalar_type = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::scalar_type;
  using index_type  = typename BinaryDataTraits<Scalar, LocalOrdinal, GlobalOrdinal>::index_type;

  auto lclA          = A.getLocalMatrixDevice();
  index_type numRows = lclA.numRows();
  index_type numCols = lclA.numCols();
  index_type nnz     = lclA.nnz();

  Kokkos::View<index_type*, Kokkos::HostSpace> rowptr_file("rowptr_file", numRows + 1);
  Kokkos::View<index_type*, Kokkos::HostSpace> colidx_file("colidx_file", nnz);
  Kokkos::View<scalar_type*, Kokkos::HostSpace> values_file("values_file", nnz);
  Kokkos::deep_copy(rowptr_file, lclA.graph.row_map);
  Kokkos::deep_copy(colidx_file, lclA.graph.entries);
  Kokkos::deep_copy(values_file, lclA.values);
  std::ofstream file(filename, std::ios::binary);
  FileType fileType = CrsMatrixFile;
  file.write(reinterpret_cast<char*>(&fileType), sizeof(FileType));
  file.write(reinterpret_cast<char*>(&numRows), sizeof(index_type));
  file.write(reinterpret_cast<char*>(&numCols), sizeof(index_type));
  file.write(reinterpret_cast<char*>(&nnz), sizeof(index_type));
  file.write(reinterpret_cast<char*>(rowptr_file.data()), (numRows + 1) * sizeof(index_type));
  file.write(reinterpret_cast<char*>(colidx_file.data()), nnz * sizeof(index_type));
  file.write(reinterpret_cast<char*>(values_file.data()), nnz * sizeof(scalar_type));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BinaryRank0ReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    BinaryRank0ReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
  : Rank0ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm, Teuchos::rcp(new BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>())) {
  // nothing
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BinaryDistributedReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    BinaryDistributedReaderWriter(const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
  : DistributedReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm, Teuchos::rcp(new BinaryLocalReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>())) {
  // nothing
}

}  // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#define TPETRA_IO_BINARY_INSTANT(SCALAR, LO, GO, NODE)          \
  template class BinaryLocalReaderWriter<SCALAR, LO, GO, NODE>; \
  template class BinaryRank0ReaderWriter<SCALAR, LO, GO, NODE>; \
  template class BinaryDistributedReaderWriter<SCALAR, LO, GO, NODE>;

#endif
