# Pull Request Checklist

Below is a simple checklist but please do not hesitate to ask for assistance!

- [ ] Fork, clone, and checkout the newest version of the code
- [ ] Create a new branch
- [ ] Make necessary code changes
- [ ] Install `ruff` (i.e., `python -m pip install ruff` or `conda install -c conda-forge ruff`)
- [ ] Install `pytest-cov` (i.e., `python -m pip install pytest-cov` or `conda install -c conda-forge pytest-cov`)
- [ ] Run `ruff check ./` in the root stumpy directory
- [ ] Run `ruff format --check --diff ./` in the root stumpy directory
- [ ] Run `./setup.sh dev && ./test.sh` in the root stumpy directory
- [ ] Reference a Github issue (and create one if one doesn't already exist)
