
Set up prerequisites:

    uv sync
    git clone git@gitlab.gwdg.de:jopitz/its-jointprobability-model.git data
    cd data
    # brew install git-lfs
    # git lfs install
    git lfs fetch
    git lfs pull
    cd ..

    DATA_DIR=$PWD/data uv run python src/its_jointprobability/webservice.py
