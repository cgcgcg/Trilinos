#include <Tpetra_Core.hpp>
#include <stdexcept>
#include <filesystem>
#include <utility>
#include "Teuchos_Assert.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include "Tpetra_IO_MatrixMarket.hpp"
#include "Tpetra_IO_Binary.hpp"

int main(int argc, char* argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  {
    std::string inputFile = "";
    std::string readMap   = "Auto";

    std::string matrixWriterType = "MatrixMarket";
    bool matrixWriterDistributed = false;
    std::string outputFile       = "";
    std::string writeMap         = "Auto";

    auto comm = Tpetra::getDefaultComm();

    Teuchos::CommandLineProcessor clp(false, true);
    clp.setOption("writer", &matrixWriterType, "");
    clp.setOption("dist", "singleRank", &matrixWriterDistributed, "");
    clp.setOption("inputFile", &inputFile, "");
    clp.setOption("readMap", &readMap, "");
    clp.setOption("outputFile", &outputFile, "");
    switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED: return EXIT_SUCCESS;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL: break;
    }

    auto out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    out->setOutputToRootOnly(0);

    *out << "Input file: " << inputFile << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(inputFile.empty(), std::runtime_error,
                               "No input file specified.");
    if (matrixWriterType != "Screen")
      TEUCHOS_TEST_FOR_EXCEPTION(outputFile.empty(), std::runtime_error,
                                 "No output file specified.");

    using Scalar        = Tpetra::Details::DefaultTypes::scalar_type;
    using LocalOrdinal  = Tpetra::Details::DefaultTypes::local_ordinal_type;
    using GlobalOrdinal = Tpetra::Details::DefaultTypes::global_ordinal_type;
    using Node          = Tpetra::Details::DefaultTypes::node_type;

    using const_map_type   = const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using multivector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using crs_matrix_type  = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto mm_proc0  = Teuchos::rcp(new Tpetra::MatrixMarketRank0ReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm));
    auto mm_dist   = Teuchos::rcp(new Tpetra::MatrixMarketDistributedReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm));
    auto bin_proc0 = Teuchos::rcp(new Tpetra::BinaryRank0ReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm));
    auto bin_dist  = Teuchos::rcp(new Tpetra::BinaryDistributedReaderWriter<Scalar, LocalOrdinal, GlobalOrdinal, Node>(comm));

    Teuchos::RCP<Tpetra::ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>> reader;
    if (mm_proc0->canReadFile(inputFile)) {
      reader = mm_proc0;
      *out << "Input file format: MatrixMarket (single rank)\n";
    } else if (mm_dist->canReadFile(inputFile)) {
      TEUCHOS_ASSERT(readMap != "None");
      reader = mm_dist;
      *out << "Input file format: MatrixMarket (distributed)\n";
    } else if (bin_proc0->canReadFile(inputFile)) {
      reader = bin_proc0;
      *out << "Input file format: Binary (single rank)\n";
    } else if (bin_dist->canReadFile(inputFile)) {
      TEUCHOS_ASSERT(readMap != "None");
      reader = bin_dist;
      *out << "Input file format: Binary (distributed)\n";
    } else
      throw std::runtime_error("No reader can read the input file");

    auto ret = reader->read(inputFile, readMap);

    if (matrixWriterType == "Screen") {
      if (ret.type() == typeid(Teuchos::RCP<const_map_type>))
        std::any_cast<Teuchos::RCP<const_map_type>>(ret)->describe(*out, Teuchos::VERB_EXTREME);
      else if (ret.type() == typeid(Teuchos::RCP<multivector_type>))
        std::any_cast<Teuchos::RCP<multivector_type>>(ret)->describe(*out, Teuchos::VERB_EXTREME);
      else if (ret.type() == typeid(Teuchos::RCP<crs_matrix_type>))
        std::any_cast<Teuchos::RCP<crs_matrix_type>>(ret)->describe(*out, Teuchos::VERB_EXTREME);
    } else {
      *out << "Output file: " << outputFile << std::endl;
      *out << "Output file format: " << matrixWriterType << (matrixWriterDistributed ? " (single rank)\n" : " (distributed)\n");

      Teuchos::RCP<Tpetra::ReaderWriterBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>> writer;
      if (!matrixWriterDistributed) {
        if (matrixWriterType == "MatrixMarket")
          writer = mm_proc0;
        else if (matrixWriterType == "Binary")
          writer = bin_proc0;
        else
          throw std::runtime_error("Unknown writer \"" + matrixWriterType + "\"");
      } else {
        if (matrixWriterType == "MatrixMarket")
          writer = mm_dist;
        else if (matrixWriterType == "Binary")
          writer = bin_dist;
        else
          throw std::runtime_error("Unknown writer \"" + matrixWriterType + "\"");
      }

      writer->write(outputFile, ret, writeMap);
    }
  }
  return EXIT_SUCCESS;
}
