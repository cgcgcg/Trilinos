#ifndef PYTRILINOS2_TPETRA_ETI
#define PYTRILINOS2_TPETRA_ETI

#include <TpetraCore_config.h>
#include <Tpetra_Core.hpp>

#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Map_def.hpp>

#include <Tpetra_Vector_decl.hpp>
#include <Tpetra_Vector_def.hpp>

#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_MultiVector_def.hpp>

#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsGraph_def.hpp>

#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_CrsMatrix_def.hpp>

#include <Tpetra_Operator.hpp>

#define BINDER_TPETRA_MAP_INSTANT(LO,GO) \
    template class Map<LO, GO>; \
    inline void initiate_map(const Map<LO, GO>& p) {};
#define BINDER_TPETRA_VECTOR_INSTANT(SCALAR,LO,GO) \
    template class Vector<SCALAR, LO, GO>; \
    inline void initiate_vector(const Vector<SCALAR, LO, GO>& p) {};
#define BINDER_TPETRA_MULTIVECTOR_INSTANT(SCALAR,LO,GO) \
    template class MultiVector<SCALAR, LO, GO>; \
    inline void initiate_multivector(const MultiVector<SCALAR, LO, GO>& p) {};

#define BINDER_TPETRA_MAP_INSTANT_2(LO,GO) \
    template class Tpetra::Map<LO, GO>; \
    template void initiate(Tpetra::Map<LO, GO> p);
#define BINDER_TPETRA_VECTOR_INSTANT_2(SCALAR,LO,GO) \
    template class Tpetra::Vector<SCALAR, LO, GO>; \
    template void initiate(Tpetra::Vector<SCALAR, LO, GO> p);
#define BINDER_TPETRA_MULTIVECTOR_INSTANT_2(SCALAR,LO,GO) \
    template class Tpetra::MultiVector<SCALAR, LO, GO>; \
    template void initiate(Tpetra::MultiVector<SCALAR, LO, GO> p);

#define BINDER_TPETRA_CRSGRAPH_INSTANT_2(LO,GO) \
    template class Tpetra::CrsGraph<LO, GO>; \
    template void initiate(Tpetra::CrsGraph<LO, GO> p);
#define BINDER_TPETRA_CRSMATRIX_INSTANT_2(SCALAR,LO,GO) \
    template class Tpetra::CrsMatrix<SCALAR, LO, GO>; \
    template void initiate(Tpetra::CrsMatrix<SCALAR, LO, GO> p);

namespace Tpetra {

    template <typename T>
    void initiate(T) {};

    BINDER_TPETRA_MAP_INSTANT_2(int, long long)
    BINDER_TPETRA_VECTOR_INSTANT_2(double, int, long long)
    BINDER_TPETRA_MULTIVECTOR_INSTANT_2(double, int, long long)

    BINDER_TPETRA_CRSGRAPH_INSTANT_2(int, long long)
    BINDER_TPETRA_CRSMATRIX_INSTANT_2(double, int, long long)

}

#endif // PYTRILINOS2_TPETRA_ETI
