{ toPythonApplication, its-jointprobability, its-jointprobability-model }:
toPythonApplication (its-jointprobability.overridePythonAttrs (prev: {
  makeWrapperArgs = [
    "--set DATA_DIR ${its-jointprobability-model}"
  ];
}))
