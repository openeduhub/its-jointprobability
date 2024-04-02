rec {
  # the strictly necessary libraries for deploying the service
  deploy = py-pkgs: with py-pkgs; [
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

  # additional libraries for (re-) training the underlying model
  train = py-pkgs: with py-pkgs; [
    optuna
    # for plotting optuna importance
    plotly
    scikit-learn
  ]
  ++ (deploy py-pkgs);

  # additional libraries for development
  devel = py-pkgs: with py-pkgs; [
    black
    pyflakes
    isort
    pylint
    ipython
    mypy
    # library stubs for mypy
    pandas-stubs
  ]
  ++ (train py-pkgs);
}
