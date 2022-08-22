rm -rf teuchos_bindings_w_stl/*
rm -rf tpetra_bindings_w_stl/*
rm -rf include_teuchos_tmp/*
rm -rf include_tpetra_tmp/*
rm -rf *_includes.hpp

mkdir include_tpetra_tmp/gtest
mkdir include_tpetra_tmp/Cuda
mkdir include_tpetra_tmp/HIP
mkdir include_tpetra_tmp/HPX
mkdir include_tpetra_tmp/setup
mkdir include_tpetra_tmp/SYCL
mkdir include_tpetra_tmp/fwd
mkdir include_tpetra_tmp/traits
mkdir include_tpetra_tmp/desul
mkdir include_tpetra_tmp/desul/atomics
mkdir include_tpetra_tmp/desul/atomics/cuda
mkdir include_tpetra_tmp/desul/atomics/openmp
mkdir include_tpetra_tmp/desul/src
mkdir include_tpetra_tmp/OpenMP
mkdir include_tpetra_tmp/OpenMPTarget
mkdir include_tpetra_tmp/Threads
mkdir include_tpetra_tmp/impl
mkdir include_tpetra_tmp/decl
mkdir include_tpetra_tmp/generated_specializations_hpp
mkdir include_tpetra_tmp/stk_util
mkdir include_tpetra_tmp/stk_util/util
mkdir include_tpetra_tmp/stk_util/parallel
mkdir include_tpetra_tmp/stk_util/environment
mkdir include_tpetra_tmp/stk_util/registry
mkdir include_tpetra_tmp/stk_util/diag
mkdir include_tpetra_tmp/stk_util/command_line
mkdir include_tpetra_tmp/stk_util/ngp
mkdir include_tpetra_tmp/stk_emend
mkdir include_tpetra_tmp/stk_emend/independent_set
mkdir include_tpetra_tmp/stk_coupling
mkdir include_tpetra_tmp/stk_math
mkdir include_tpetra_tmp/stk_simd
mkdir include_tpetra_tmp/stk_simd/kokkos_simd
mkdir include_tpetra_tmp/stk_simd_view
mkdir include_tpetra_tmp/stk_ngp_test
mkdir include_tpetra_tmp/stk_topology
mkdir include_tpetra_tmp/stk_topology/topology_detail
mkdir include_tpetra_tmp/stk_mesh
mkdir include_tpetra_tmp/stk_mesh/base
mkdir include_tpetra_tmp/stk_mesh/baseImpl
mkdir include_tpetra_tmp/stk_mesh/baseImpl/elementGraph
mkdir include_tpetra_tmp/stk_io
mkdir include_tpetra_tmp/stk_io/util
mkdir include_tpetra_tmp/stk_search
mkdir include_tpetra_tmp/stk_transfer
mkdir include_tpetra_tmp/stk_transfer/copy_by_id
mkdir include_tpetra_tmp/stk_tools
mkdir include_tpetra_tmp/stk_tools/mesh_clone
mkdir include_tpetra_tmp/stk_tools/mesh_tools
mkdir include_tpetra_tmp/stk_tools/block_extractor
mkdir include_tpetra_tmp/stk_tools/transfer_utils
mkdir include_tpetra_tmp/stk_balance
mkdir include_tpetra_tmp/stk_balance/internal
mkdir include_tpetra_tmp/stk_balance/m2n
mkdir include_tpetra_tmp/stk_balance/setup
mkdir include_tpetra_tmp/stk_balance/search_tolerance
mkdir include_tpetra_tmp/stk_balance/search_tolerance_algs
mkdir include_tpetra_tmp/stk_unit_test_utils
mkdir include_tpetra_tmp/stk_unit_test_utils/stk_mesh_fixtures
mkdir include_tpetra_tmp/stk_expreval
mkdir include_tpetra_tmp/std_algorithms
mkdir include_tpetra_tmp/std_algorithms/modifying_sequence_ops
mkdir include_tpetra_tmp/std_algorithms/numeric


python gather_includes.py
cp include/Teuchos_Include_Pybind11.hpp include_tpetra_tmp/.
cp include/Tpetra_Include_Pybind11.hpp include_tpetra_tmp/.
cp include/Teuchos_Include_Pybind11.cpp tpetra_bindings_w_stl/.

$HOMEBINDER/prefix/build/bin/binder -v \
  --root-module PyTrilinos2 \
  --prefix $PWD/tpetra_bindings_w_stl -max-file-size=1000000\
  --bind Teuchos --bind Tpetra --config PyTrilinos2_config.cfg \
  tpetra_includes.hpp \
  -- -std=c++17 -I$PWD/include_tpetra_tmp -I$PWD/include_tpetra_tmp/impl \
  --gcc-toolchain="/projects/sems/install/rhel7-x86_64/sems-compilers/tpl/gcc/7.3.0/gcc/4.8.5/base/47wbtzd" \
  -DNDEBUG

echo "Teuchos_Include_Pybind11.cpp" >> tpetra_bindings_w_stl/PyTrilinos2.sources
