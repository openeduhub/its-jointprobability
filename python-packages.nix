rec {
  # the strictly necessary libraries for deploying the service
  deploy-pkgs = py-pkgs: with py-pkgs; [
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
  ];

  # additional libraries for hyperparameter optimization of the model
  optuna-pkgs = py-pkgs: with py-pkgs; [
    optuna
    # for plotting optuna importance
    plotly
    scikit-learn
  ]
  ++ (deploy-pkgs py-pkgs);

  # additional libraries for development
  devel-pkgs = py-pkgs: with py-pkgs; [
    black
    pyflakes
    isort
    pylint
    ipython
    mypy
    # library stubs for mypy
    pandas-stubs
  ]
  ++ (optuna-pkgs py-pkgs);
}