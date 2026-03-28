from cmaketracing import main
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    # The instrumentation feature will pass the path to an index file to our callback
    index = Path(sys.argv[1])
    assert index.exists()

    # Get the buildDir and dataDir from the index file
    with open(index) as f:
        data = json.load(f)
    buildDir = Path(data["buildDir"])
    dataDir = data["dataDir"]

    # Get a unique output directory name based on the index filename
    indexName = index.name.split(".")[0]

    # Create an output directory that CMake won't clear, to copy our files into
    outputDir = buildDir/"instrumentation"
    indexDir = outputDir/indexName
    newDataDir = indexDir/"data"
    newDataDir.mkdir(parents=True, exist_ok=True)

    # Copy all the instrumentation data into our output directory
    for snippet in data["snippets"]:
        src = Path(dataDir)/snippet
        assert src.exists()
        tgt = newDataDir/snippet
        src.rename(tgt)

    data["dataDir"] = str(newDataDir)
    newIndex = indexDir/index.name
    with open(newIndex, "w") as f:
        json.dump(data, f)

    # Generate our trace.json file
    main(newIndex, indexDir/"trace.json")
