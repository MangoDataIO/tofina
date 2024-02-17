rm -rf ./tests/results
mkdir ./tests/results
python -m pytest tests --ignore=tests/integration

