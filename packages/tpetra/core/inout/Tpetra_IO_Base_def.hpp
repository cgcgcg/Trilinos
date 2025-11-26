#ifndef TPETRA_IO_BASE_DEF_HPP
#define TPETRA_IO_BASE_DEF_HPP

#include "Teuchos_Assert.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TestForException.hpp"
#include "Tpetra_IO_Base_decl.hpp"
#include <filesystem>

namespace Tpetra {

namespace {
ReadWriteMaps stringToEnum(std::string str) {
  std::map<std::string, Tpetra::ReadWriteMaps> optionMap{
      std::make_pair("None", Tpetra::None),
      std::make_pair("Auto", Tpetra::Auto),
      std::make_pair("All", Tpetra::All)};
  return optionMap[str];
}

}  // namespace

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::vector_type>
LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readVector(const std::string& filename) const {
  return Teuchos::rcp_dynamic_cast<vector_type>(readMultiVector(filename), true);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::write(const std::string& filename,
                                                                             const vector_type& v) const {
  write(filename, static_cast<multivector_type>(v));
}
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const typename LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_type>
LocalReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildLocalMap(LocalOrdinal numUnknowns) const {
  auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_SELF));
  auto map  = Teuchos::rcp(new map_type((GlobalOrdinal)numUnknowns, numUnknowns, 0, comm));
  return map;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ReaderWriterBase(const Teuchos::RCP<const Teuchos::Comm<int>>& comm) {
  comm_ = comm;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Teuchos::Comm<int>> ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getComm() const {
  return comm_;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::multivector_type>
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readMultiVector(const std::string& filename,
                    ReadWriteMaps readMapFile) const {
  std::string mapFilename = filename + ".map";
  if (readMapFile == None) {
    return readMultiVector(filename, Teuchos::null);
  } else if (readMapFile == Auto) {
    Teuchos::RCP<const map_type> map;
    if (std::filesystem::exists(mapFilename))
      map = readMap(mapFilename);
    return readMultiVector(filename, map);
  } else if (readMapFile == All) {
    auto map = readMap(mapFilename);
    return readMultiVector(filename, map);
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::multivector_type>
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readMultiVector(const std::string& filename,
                    std::string readMapFile) const {
  return readMultiVector(filename, stringToEnum(readMapFile));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type>
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readCrsMatrix(const std::string& filename,
                  ReadWriteMaps readMaps) const {
  std::string rowmapFilename    = filename + ".rowmap";
  std::string domainmapFilename = filename + ".domainmap";
  std::string rangemapFilename  = filename + ".rangemap";
  std::string colmapFilename    = filename + ".colmap";

  Teuchos::RCP<const map_type> rowmap;
  Teuchos::RCP<const map_type> colmap;
  Teuchos::RCP<const map_type> domainmap;
  Teuchos::RCP<const map_type> rangemap;
  if (readMaps == None) {
    // Nothing to do
  } else if (readMaps == Auto) {
    if (std::filesystem::exists(rowmapFilename))
      rowmap = readMap(rowmapFilename);
    if (std::filesystem::exists(colmapFilename))
      colmap = readMap(colmapFilename);
    if (std::filesystem::exists(domainmapFilename))
      domainmap = readMap(domainmapFilename);
    if (std::filesystem::exists(rangemapFilename))
      rangemap = readMap(rangemapFilename);
  } else if (readMaps == All) {
    rowmap    = readMap(rowmapFilename);
    colmap    = readMap(colmapFilename);
    domainmap = readMap(domainmapFilename);
    rangemap  = readMap(rangemapFilename);
  }
  return readCrsMatrix(filename, rowmap, colmap, domainmap, rangemap);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type>
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    readCrsMatrix(const std::string& filename,
                  std::string readMapFiles) const {
  return readCrsMatrix(filename, stringToEnum(readMapFiles));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::any
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    read(const std::string& filename,
         ReadWriteMaps readMapFiles) const {
  auto fileType = detectFileType(filename);
  if (fileType == MapFile) {
    auto map     = readMap(filename);
    std::any ret = map;
    return ret;
  } else if (fileType == MultiVectorFile) {
    auto mv      = readMultiVector(filename, readMapFiles);
    std::any ret = mv;
    return ret;
  } else if (fileType == CrsMatrixFile) {
    auto crs     = readCrsMatrix(filename, readMapFiles);
    std::any ret = crs;
    return ret;
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::any
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    read(const std::string& filename,
         std::string readMapFiles) const {
  return read(filename, stringToEnum(readMapFiles));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const multivector_type& mv,
          ReadWriteMaps writeMap) const {
  write(filename, mv);
  if ((writeMap == Auto) || (writeMap == All)) {
    std::string mapFilename = filename + ".map";
    write(mapFilename, *mv.getMap());
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const multivector_type& mv,
          std::string writeMap) const {
  write(filename, mv, stringToEnum(writeMap));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const crs_matrix_type& crs,
          ReadWriteMaps writeMaps) const {
  write(filename, crs);

  std::string rowmapFilename    = filename + ".rowmap";
  std::string domainmapFilename = filename + ".domainmap";
  std::string rangemapFilename  = filename + ".rangemap";
  std::string colmapFilename    = filename + ".colmap";
  if (writeMaps == None) {
    // Nothing to do
  } else if (writeMaps == Auto) {
    write(rowmapFilename, *crs.getRowMap());
    if (!crs.getRowMap()->isSameAs(*crs.getRangeMap()))
      write(rangemapFilename, *crs.getRangeMap());
    if (!crs.getRangeMap()->isSameAs(*crs.getDomainMap()))
      write(domainmapFilename, *crs.getDomainMap());
    if (!crs.getDomainMap()->isSameAs(*crs.getColMap()))
      write(colmapFilename, *crs.getColMap());
  } else if (writeMaps == All) {
    write(rowmapFilename, *crs.getRowMap());
    write(colmapFilename, *crs.getColMap());
    write(rangemapFilename, *crs.getRangeMap());
    write(domainmapFilename, *crs.getDomainMap());
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const crs_matrix_type& crs,
          std::string writeMap) const {
  write(filename, crs, stringToEnum(writeMap));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const std::any& obj,
          ReadWriteMaps writeMaps) const {
  if (obj.type() == typeid(const map_type))
    write(filename, std::any_cast<const map_type>(obj));
  else if (obj.type() == typeid(multivector_type))
    write(filename, std::any_cast<multivector_type>(obj), writeMaps);
  else if (obj.type() == typeid(crs_matrix_type))
    write(filename, std::any_cast<crs_matrix_type>(obj), writeMaps);
  else if (obj.type() == typeid(Teuchos::RCP<const map_type>))
    write(filename, *std::any_cast<Teuchos::RCP<const map_type>>(obj));
  else if (obj.type() == typeid(Teuchos::RCP<multivector_type>))
    write(filename, *std::any_cast<Teuchos::RCP<multivector_type >> (obj), writeMaps);
  else if (obj.type() == typeid(Teuchos::RCP<crs_matrix_type>))
    write(filename, *std::any_cast<Teuchos::RCP<crs_matrix_type>>(obj), writeMaps);
  else
    throw std::runtime_error(std::string("Unhandled type: ") + obj.type().name());
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    write(const std::string& filename,
          const std::any& obj,
          std::string writeMap) const {
  write(filename, obj, stringToEnum(writeMap));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<typename ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_multivector_type>
ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    mapToMultiVector(const map_type& map) const {
  auto map_mv = Teuchos::rcp(new map_multivector_type(Teuchos::rcpFromRef(map), 2, false));
  auto comm   = getComm();
  {
    auto gids_procs = map_mv->getLocalViewDevice(Access::OverwriteAll);
    Kokkos::deep_copy(Kokkos::subview(gids_procs, Kokkos::ALL(), 0), map.getMyGlobalIndicesDevice());
    Kokkos::deep_copy(Kokkos::subview(gids_procs, Kokkos::ALL(), 1), comm->getRank());
  }
  return map_mv;
}

}  // namespace Tpetra

#endif
