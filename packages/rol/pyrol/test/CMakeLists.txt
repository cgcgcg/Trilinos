ENABLE_TESTING()

ADD_SUBDIRECTORY(algorithm)
ADD_SUBDIRECTORY(unsupported)

INCLUDE(PythonTest)

PYTHON_ADD_TEST(test_tolerance)
PYTHON_ADD_TEST(test_tolerance2)
