{
  lib,
  nix-filter,
  buildPythonPackage,
  setuptools,
  pandas,
  uvicorn,
  pydantic,
  fastapi,
  pyro-ppl,
  icecream,
  matplotlib,
  tqdm,
  its-data,
  optuna,
  plotly,
  scikit-learn,
  sqlite,
  its-jointprobability-model,
  withOptuna ? false,
}:
buildPythonPackage {
  pname = "its-jointprobability";
  version = "0.2.1";
  format = "setuptools";

  src = nix-filter {
    root = ./.;
    # only include files that are related to the application.
    # this will prevent unnecessary rebuilds
    include = [
      (nix-filter.inDirectory ./its_jointprobability)
      ./setup.py
      ./requirements.txt
    ];
    exclude =
      [ (nix-filter.matchExt "pyc") ]
      # optionally ignore hyperparameter-optimization related code
      ++ (lib.lists.optionals (!withOptuna) [ (nix-filter.inDirectory ./its_jointprobability/optuna) ]);
  };

  # replace local lookups of the model
  prePatch = ''
    substituteInPlace its_jointprobability/*.py \
      --replace "Path.cwd() / \"data\"" \
                "Path(\"${its-jointprobability-model}\")"
  '';

  propagatedBuildInputs =
    [
      setuptools
      pandas
      uvicorn
      pydantic
      fastapi
      pyro-ppl
      icecream
      matplotlib
      tqdm
      its-data
    ]
    ++ (lib.lists.optionals withOptuna [
      optuna
      plotly
      scikit-learn
      sqlite
    ]);
}
