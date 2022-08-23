FILE(GLOB PYTRILINOS2_SRC *.cpp)
MESSAGE("PYTRILINOS2_SRC = ${PYTRILINOS2_SRC}")
FILE(COPY ${PYTRILINOS2_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/../binder/PyTrilinos2.sources BINDER_SRCS)
list(TRANSFORM BINDER_SRCS PREPEND "../binder/")
list(APPEND PYTRILINOS2_SRC ${BINDER_SRCS})

MESSAGE("PYTRILINOS2_SRC with binder = ${PYTRILINOS2_SRC}")

pybind11_add_module(PyTrilinos2 ${PYTRILINOS2_SRC})
target_include_directories(PyTrilinos2 PUBLIC ${Mpi4Py_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PyTrilinos2 PUBLIC cxx_std_14)

foreach(depPkg IN LISTS PyTrilinos2_LIB_ENABLED_DEPENDENCIES) 
    target_link_libraries(PyTrilinos2 PUBLIC ${depPkg}::all_libs)
endforeach()
set_target_properties(PyTrilinos2 PROPERTIES SUFFIX ".so")
