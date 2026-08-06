// @HEADER
// *****************************************************************************
//                 Belos: Block Linear Solvers Package
//
// Copyright 2004-2016 NTESS and the Belos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
//  This test exercises the Belos dense matrix solvers (TeuchosDenseSolver and
//  KokkosDenseSolver) through the DenseSolver / DenseMatTraits interfaces.
//  It checks the correctness of factor()/solve() for general and SPD systems,
//  with and without equilibration, and with transpose solves, by computing the
//  residual of the solution against the unmodified original system.
//

#include "BelosConfigDefs.hpp"
#include "BelosTypes.hpp"
#include "BelosOutputManager.hpp"
#include "BelosDenseMatTraits.hpp"
#include "BelosDenseSolver.hpp"
#include "BelosTeuchosDenseAdapter.hpp"
#include "BelosKokkosDenseAdapter.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <vector>
#include <string>
#include <iostream>

#ifdef HAVE_TEUCHOS_COMPLEX
#include <complex>
#endif

#ifdef HAVE_MPI
  #include <mpi.h>
#endif

namespace Belos {

  template<class ScalarType, class DM>
  void fillMatrix(DM& dm, const std::vector<ScalarType>& values) {
    typedef DenseMatTraits<ScalarType,DM> DMT;
    const int M = DMT::GetNumRows(dm);
    const int N = DMT::GetNumCols(dm);
    for (int j=0; j<N; ++j) {
      for (int i=0; i<M; ++i) {
        DMT::Value(dm,i,j) = values[j*M + i];
      }
    }
  }

  //! Solve the given system and return the relative residual \|A X - B\| / \|B\|.
  template<class ScalarType, class DM>
  bool solveAndCheck(const Teuchos::RCP<OutputManager<ScalarType> >& om,
                     const std::vector<ScalarType>& Aorig,
                     const std::vector<ScalarType>& Borig,
                     const int n, const int nrhs,
                     const bool spd, const Teuchos::ETransp trans,
                     const bool equilibrate, const char* label)
  {
    using Teuchos::RCP;
    using Teuchos::rcp;
    typedef DenseMatTraits<ScalarType,DM> DMT;
    typedef Teuchos::ScalarTraits<ScalarType> STS;
    typedef typename STS::magnitudeType Mag;
    typedef Teuchos::ScalarTraits<Mag> MSTS;

    // Fresh copies of A and B, since factorization overwrites the matrix
    // in place.  X is initialized with B so that the solver solves in place.
    RCP<DM> A = DMT::Create(n, n);
    fillMatrix<ScalarType,DM>(*A, Aorig);
    RCP<DM> X = DMT::Create(n, nrhs);
    fillMatrix<ScalarType,DM>(*X, Borig);

    RCP<DenseSolver<ScalarType,DM> > solver = DMT::createDenseSolver();
    solver->setMatrix(A);
    solver->setVectors(X, X);
    solver->setSPD(spd);
    solver->solveWithTransposeFlag(trans);
    solver->factorWithEquilibration(equilibrate);

    int info = solver->factor();
    if (info != 0) {
      om->stream(Warnings) << "*** ERROR *** DenseSolver factor failed (info="
                           << info << ") for case \"" << label << "\"." << std::endl;
      return false;
    }
    info = solver->solve();
    if (info != 0) {
      om->stream(Warnings) << "*** ERROR *** DenseSolver solve failed (info="
                           << info << ") for case \"" << label << "\"." << std::endl;
      return false;
    }

    // Compute the residual \|A X - B\|_F / \|B\|_F using the original A.
    Mag resid = MSTS::zero();
    Mag bnorm = MSTS::zero();
    for (int j=0; j<nrhs; ++j) {
      for (int i=0; i<n; ++i) {
        ScalarType s = STS::zero();
        for (int k=0; k<n; ++k) {
          const ScalarType aij = (trans == Teuchos::NO_TRANS) ?
            Aorig[k*n + i] : Aorig[i*n + k];
          s += aij * DMT::ValueConst(*X,k,j);
        }
        s -= Borig[j*n + i];
        resid += STS::magnitude(s)*STS::magnitude(s);
        const Mag bmag = STS::magnitude(Borig[j*n + i]);
        bnorm += bmag*bmag;
      }
    }
    resid = MSTS::squareroot(resid);
    bnorm = MSTS::squareroot(bnorm);

    const Mag relResid = resid / bnorm;
    const Mag tol = Mag(500.0) * MSTS::eps() * Mag(n*nrhs);

    if (relResid > tol) {
      om->stream(Warnings) << "*** ERROR *** DenseSolver case \"" << label
                           << "\" had relative residual " << relResid
                           << " (tol " << tol << ")." << std::endl;
      return false;
    }
    return true;
  }

  template<class ScalarType, class DM>
  bool testDenseSolver(const Teuchos::RCP<OutputManager<ScalarType> >& om)
  {
    using Teuchos::NO_TRANS;
    using Teuchos::TRANS;
    typedef Teuchos::ScalarTraits<ScalarType> STS;

    const int n = 8;
    const int nrhs = 3;

    // General nonsymmetric, strictly diagonally dominant matrix.
    std::vector<ScalarType> Aorig(n*n);
    for (int j=0; j<n; ++j) {
      for (int i=0; i<n; ++i) {
        if (i == j)
          Aorig[j*n + i] = ScalarType(100.0 + 3*i);
        else
          Aorig[j*n + i] = ScalarType((i*13 + j*7) % 19) - ScalarType(9.0);
      }
    }
    std::vector<ScalarType> Borig(n*nrhs);
    for (int j=0; j<nrhs; ++j) {
      for (int i=0; i<n; ++i) {
        Borig[j*n + i] = ScalarType((i*3 + j*5) % 7) - ScalarType(2.0);
      }
    }

    bool success = true;

    // General system: with/without equilibration, normal/transpose solve.
    for (int eq = 0; eq <= 1; ++eq) {
      for (int trans = 0; trans <= 1; ++trans) {
        std::string label = (eq ? "equilibrated " : "") +
                            std::string(trans ? "transposed" : "general");
        const bool ok = solveAndCheck<ScalarType,DM>(om, Aorig, Borig, n, nrhs,
                                                     false, trans ? TRANS : NO_TRANS,
                                                     eq == 1, label.c_str());
        if (!ok) success = false;
      }
    }

    // SPD system: only meaningful for real scalar types.
    if (!STS::isComplex) {
      // A = M^T M + (n+1) I, with random M.
      std::vector<ScalarType> Mvals(n*n);
      for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
          Mvals[j*n + i] = ScalarType(((i*7 + j*3) % 11) - 5);
        }
      }
      std::vector<ScalarType> Aspd(n*n, STS::zero());
      for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
          ScalarType s = STS::zero();
          for (int k=0; k<n; ++k) {
            s += Mvals[k*n + i]*Mvals[k*n + j];
          }
          Aspd[j*n + i] = s + ScalarType(n+1)*(i == j ? ScalarType(1.0) : ScalarType(0.0));
        }
      }
      for (int eq = 0; eq <= 1; ++eq) {
        std::string label = std::string("SPD ") +
                            std::string(eq ? "(equilibrated)" : "(not equilibrated)");
        const bool ok = solveAndCheck<ScalarType,DM>(om, Aspd, Borig, n, nrhs,
                                                     true, NO_TRANS, eq == 1,
                                                     label.c_str());
        if (!ok) success = false;
      }
    }

    return success;
  }

} // namespace Belos

namespace {

  template<class ScalarType, class DM>
  bool runScalarTest(const std::string& name, const bool verbose)
  {
    using Teuchos::RCP;
    using Teuchos::rcp;
    RCP<Belos::OutputManager<ScalarType> > om =
      rcp(new Belos::OutputManager<ScalarType>(verbose ? Belos::Errors
                                                       : Belos::Warnings));
    const bool ok = Belos::testDenseSolver<ScalarType,DM>(om);
    std::cout << "DenseSolver test for " << name << ": "
              << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
  }

} // namespace

int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  bool success = true;
  Kokkos::initialize();

  {
    bool verbose = false;
    if (argc > 1) {
      if (argv[1][0]=='-' && argv[1][1]=='v') {
        verbose = true;
      }
    }

    typedef Teuchos::SerialDenseMatrix<int, double> SDM_double;
    typedef Teuchos::SerialDenseMatrix<int, float> SDM_float;
    typedef Kokkos::DualView<typename KokkosKernels::ArithTraits<double>::val_type**, Kokkos::LayoutLeft> DVT_double;
    typedef Kokkos::DualView<typename KokkosKernels::ArithTraits<float>::val_type**, Kokkos::LayoutLeft> DVT_float;

    if (!runScalarTest<double, SDM_double>("TeuchosDenseSolver<double>", verbose))
      success = false;
    if (!runScalarTest<double, DVT_double>("KokkosDenseSolver<double>", verbose))
      success = false;
    if (!runScalarTest<float, SDM_float>("TeuchosDenseSolver<float>", verbose))
      success = false;
    if (!runScalarTest<float, DVT_float>("KokkosDenseSolver<float>", verbose))
      success = false;

#ifdef HAVE_TEUCHOS_COMPLEX
    typedef Teuchos::SerialDenseMatrix<int, std::complex<double>> SDM_complex_double;
    typedef Kokkos::DualView<typename KokkosKernels::ArithTraits<std::complex<double>>::val_type**, Kokkos::LayoutLeft> DVT_complex_double;
    if (!runScalarTest<std::complex<double>, SDM_complex_double>("TeuchosDenseSolver<complex<double>>", verbose))
      success = false;
    if (!runScalarTest<std::complex<double>, DVT_complex_double>("KokkosDenseSolver<complex<double>>", verbose))
      success = false;
#endif
  }

  Kokkos::finalize();
#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if (!success) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
