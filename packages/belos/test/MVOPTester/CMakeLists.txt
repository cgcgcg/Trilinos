

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)
IF (${PACKAGE_NAME}_ENABLE_Triutils)
    
  ASSERT_DEFINED(Teuchos_ENABLE_COMPLEX)
  IF(Teuchos_ENABLE_COMPLEX)

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      MVOPComplexTester
      SOURCES cxx_main_complex.cpp MyMultiVec.hpp MyOperator.hpp MyBetterOperator.hpp
      COMM serial mpi
      ARGS
        "--verbose --filename=mhd1280b.cua"
      STANDARD_PASS_OUTPUT
      )

    TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyTestMVOPTesterComplexFiles
      SOURCE_DIR ${Belos_SOURCE_DIR}/testmatrices
      SOURCE_FILES mhd1280b.cua 
      EXEDEPS MVOPComplexTester
      )

  ENDIF(Teuchos_ENABLE_COMPLEX)

ENDIF(${PACKAGE_NAME}_ENABLE_Triutils)
