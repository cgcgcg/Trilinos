INCLUDE(TrilinosCreateClientTemplateHeaders)

FUNCTION(GALERI_PROCESS_ETI_TEMPLATE_SLGN ETI_CLASSES TEMPLATE_FILE SOURCES_LIST GALERI_ETI_SCALARS GALERI_ETI_LORDS GALERI_ETI_GORDS GALERI_ETI_NODES)
  SET(SRCS "")
  FOREACH(CLASS ${ETI_CLASSES})
    TPETRA_MANGLE_TEMPLATE_PARAMETER(CLASS_MANGLED ${CLASS})
    string(TOUPPER "${CLASS_MANGLED}" UPPER_CASE_CLASS)
    TPETRA_PROCESS_ALL_SLGN_TEMPLATES(TMP_OUTPUT_FILES ${TEMPLATE_FILE} ${CLASS_MANGLED} ${UPPER_CASE_CLASS} "${GALERI_ETI_SCALARS}" "${GALERI_ETI_LORDS}" "${GALERI_ETI_GORDS}" "${GALERI_ETI_NODES}" FALSE)
    LIST(APPEND SRCS ${TMP_OUTPUT_FILES})
  ENDFOREACH()
  SET(${SOURCES_LIST} ${SRCS} PARENT_SCOPE)
ENDFUNCTION(GALERI_PROCESS_ETI_TEMPLATE_SLGN)

FUNCTION(GALERI_PROCESS_ETI_TEMPLATE_LGN ETI_CLASSES TEMPLATE_FILE SOURCES_LIST GALERI_ETI_LORDS GALERI_ETI_GORDS GALERI_ETI_NODES)
  SET(SRCS "")
  FOREACH(CLASS ${ETI_CLASSES})
    TPETRA_MANGLE_TEMPLATE_PARAMETER(CLASS_MANGLED ${CLASS})
    string(TOUPPER "${CLASS_MANGLED}" UPPER_CASE_CLASS)
    TPETRA_PROCESS_ALL_LGN_TEMPLATES(TMP_OUTPUT_FILES ${TEMPLATE_FILE} ${CLASS_MANGLED} ${UPPER_CASE_CLASS} "${GALERI_ETI_LORDS}" "${GALERI_ETI_GORDS}" "${GALERI_ETI_NODES}")
    LIST(APPEND SRCS ${TMP_OUTPUT_FILES})
  ENDFOREACH()
  SET(${SOURCES_LIST} ${SRCS} PARENT_SCOPE)
ENDFUNCTION(GALERI_PROCESS_ETI_TEMPLATE_LGN)


# Make sure that Tpetra actually defined these variables, even if they
# are empty.
ASSERT_DEFINED(TpetraCore_ETI_SCALARS_NO_ORDS)
ASSERT_DEFINED(TpetraCore_ETI_LORDS)
ASSERT_DEFINED(TpetraCore_ETI_GORDS)
ASSERT_DEFINED(TpetraCore_ETI_NODES)

SET(Galeri_ETI_SCALARS ${TpetraCore_ETI_SCALARS_NO_ORDS})
SET(Galeri_ETI_LORDS   ${TpetraCore_ETI_LORDS})
SET(Galeri_ETI_GORDS   ${TpetraCore_ETI_GORDS})
SET(Galeri_ETI_NODES   ${TpetraCore_ETI_NODES})


TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

SET(HEADERS "")
SET(SOURCES "")

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src-epetra)
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/../src-epetra)
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../headers)


APPEND_SET(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
)

APPEND_SET(HEADERS
  ../headers/Galeri_ConfigDefs.h
  ../headers/Galeri_Exception.h

  Galeri_XpetraParameters.hpp

  Galeri_XpetraProblemFactory.hpp
  Galeri_XpetraProblemFactory_Helmholtz.hpp
  Galeri_XpetraMatrixTypes.hpp
  Galeri_XpetraMatrixTypes_Helmholtz.hpp
  Galeri_StencilProblems.hpp
  Galeri_StencilProblems_Helmholtz.hpp
  Galeri_Elasticity2DProblem.hpp
  Galeri_Elasticity3DProblem.hpp
  Galeri_Problem_Helmholtz.hpp
  Galeri_HelmholtzFEM2DProblem.hpp
  Galeri_HelmholtzFEM3DProblem.hpp
  Galeri_VelocityModel.hpp

  Galeri_VectorTraits.hpp
  Galeri_MatrixTraits.hpp
  Galeri_MapTraits.hpp
  Galeri_MultiVectorTraits.hpp

  Galeri_XpetraUtils.hpp
  Galeri_Problem.hpp

  Galeri_XpetraMaps.hpp
  Galeri_XpetraCartesian.hpp
)

APPEND_SET(SOURCES
  mac.cpp
  )

TRILINOS_CREATE_CLIENT_TEMPLATE_HEADERS(${CMAKE_CURRENT_SOURCE_DIR})

# Automatically generate ETI (explicit template instanatiation) files
# for Node types that use the (new) Kokkos Devices.
SET(GALERI_CPP_SOURCES "")
IF(Galeri_ENABLE_EXPLICIT_INSTANTIATION)
  # Set the list of Galeri classes templated on <Scalar, LO, GO,
  # Node> for which we want to do ETI using this system.  These
  # classes usually operate on sparse matrices (instances of
  # Tpetra::CrsMatrix or Tpetra::RowMatrix, which also take these
  # template parameters).
  GLOBAL_SET(Galeri_ETI_CLASSES_SLGN
    StencilProblems
    XpetraProblemFactory
    XpetraUtils
  )

  GLOBAL_SET(Galeri_ETI_CLASSES_LGN
    XpetraCartesian
    XpetraMaps
  )


  SET(TEMPLATE_FILE "Galeri_ETI_SC_LO_GO_NT.tmpl")
  GALERI_PROCESS_ETI_TEMPLATE_SLGN("${Galeri_ETI_CLASSES_SLGN}" ${TEMPLATE_FILE} Galeri_SRCS "${Galeri_ETI_SCALARS}" "${Galeri_ETI_LORDS}" "${Galeri_ETI_GORDS}" "${Galeri_ETI_NODES}")
  LIST(APPEND Galeri_CPP_SOURCES ${Galeri_SRCS})

  SET(TEMPLATE_FILE "Galeri_ETI_LO_GO_NT.tmpl")
  GALERI_PROCESS_ETI_TEMPLATE_LGN("${Galeri_ETI_CLASSES_LGN}" ${TEMPLATE_FILE} Galeri_SRCS "${Galeri_ETI_LORDS}" "${Galeri_ETI_GORDS}" "${Galeri_ETI_NODES}")
  LIST(APPEND Galeri_CPP_SOURCES ${Galeri_SRCS})

ENDIF()

TRIBITS_ADD_LIBRARY(
  galeri-xpetra
  HEADERS ${HEADERS}
  SOURCES ${SOURCES} ${Galeri_CPP_SOURCES}
)
