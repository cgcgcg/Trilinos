#include <mpi.h>

// #include "Teuchos_DefaultComm.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
// #include "Teuchos_Comm.hpp"

#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_BlockVector.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_BlockCrsMatrix.hpp"
#include "Tpetra_Operator.hpp"
#include <Tpetra_Version.hpp> // Tpetra version string

#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosSolverFactory.hpp"
#include "BelosSolverManager.hpp"

void doStuff(Teuchos::RCP<const Teuchos::Comm<int> > teuchosComm){

    // typedefs
    using LO = Tpetra::Vector<>::local_ordinal_type;
    using GO = Tpetra::Vector<>::global_ordinal_type;
    using MapType              = Tpetra::Map<LO, GO>;
    using CrsGraphType         = Tpetra::CrsGraph<LO, GO>;
    using CrsMatrixType        = Tpetra::CrsMatrix<double, LO, GO>;
    using CrsMatrixPointerType = Teuchos::RCP<CrsMatrixType>;
    using VectorType           = Tpetra::Vector<double, LO, GO>;
    using SystemMatrix         = Tpetra::CrsMatrix<double, LO, GO, Tpetra::Map<>::node_type>;
    using Scalar               = SystemMatrix::scalar_type;
    using MultiVectorType      = Tpetra::MultiVector<Scalar, LO, GO>;
    using OperatorType         = Tpetra::Operator<Scalar, LO, GO>;
    using LinearProblemType    = Belos::LinearProblem<Scalar, MultiVectorType, OperatorType>;

    // pointMap and pointCrsGraph
    GO indexBase = 0;
    std::vector<GO> myGlobalPointRows;
    std::vector<long unsigned int> numEntriesPerRow;
    auto pointMap = Teuchos::rcp(new MapType( 20, Teuchos::arrayViewFromVector(myGlobalPointRows), indexBase, teuchosComm));
    auto pointCrsGraph = Teuchos::rcp(new CrsGraphType( pointMap, Teuchos::arrayViewFromVector(numEntriesPerRow)));
    pointCrsGraph->fillComplete();

    // systemMatrix
    CrsMatrixPointerType systemMatrix;
    systemMatrix = Teuchos::rcp(new SystemMatrix(pointCrsGraph));
    systemMatrix->fillComplete();

    // source and solution vectors
    auto sourceVector   = Teuchos::rcp(new VectorType(systemMatrix->getMap()));
    auto solutionVector = Teuchos::rcp(new VectorType(systemMatrix->getMap()));

    // solver variables
    Belos::SolverFactory<Scalar, MultiVectorType, OperatorType> solverFactory;
    Teuchos::RCP<Belos::SolverManager<Scalar, MultiVectorType, OperatorType> > solverManager;
    Teuchos::RCP<LinearProblemType> linearProblem;
    const Teuchos::RCP<Teuchos::ParameterList> solverParams;

    // solver execution
    solverManager = solverFactory.create("GMRES", solverParams);
    linearProblem = Teuchos::rcp(new LinearProblemType(Teuchos::RCP<const Tpetra::RowMatrix<Scalar, LO, GO>>(systemMatrix), solutionVector, sourceVector));
    linearProblem->setProblem(); // <-- segfaults here
    solverManager->setProblem(linearProblem);
    solverManager->solve();
}

int main (int argc, char* argv[])
{
    // Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm>> rawComm = Teuchos::opaqueWrapper<MPI_Comm>(MPI_COMM_WORLD);
    // Teuchos::RCP<Teuchos::Comm<int>> teuchosComm = Teuchos::createMpiComm<int>(rawComm);

    (void) MPI_Init (&argc, &argv);
    MPI_Comm yourComm = MPI_COMM_WORLD;
    Teuchos::RCP<const Teuchos::Comm<int> > teuchosComm = Teuchos::rcp (new Teuchos::MpiComm<int> (yourComm));

    doStuff(teuchosComm);

    (void) MPI_Finalize ();
    return 0;
}
