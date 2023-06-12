default: pytest

# default: pylint pytest

# pylint:
# 	find . -iname "*.py" -not -path "./tests/test_*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	echo "no tests"

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------


run_pred:
	python -c 'from RNN_MUSIC_GENERATOR.Generator.melody_generator import predict; predict()'

run_evaluate:
	python -c 'from main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate


run_upload:
	python -c 'from RNN_MUSIC_GENERATOR.Interface.load_model_mlflow import upload_model; upload_model()'

run_api:
	uvicorn RNN_MUSIC_GENERATOR.Generator.melody_generator:app --reload
