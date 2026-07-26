# Pull Request Checklist

Below is a simple checklist but please do not hesitate to ask for assistance!

- [ ] Read our [Contributing Guide](https://stumpy.readthedocs.io/en/latest/Contribute.html)
- [ ] Referenced a Github issue (or create one if one doesn't already exist)
- [ ] Left a meaningful comment on the original Github issue to discuss the detailed approach for your contribution
- [ ] Forked, cloned, and checkedout the newest version of the code
- [ ] Created a new branch
- [ ] Made necessary code changes
- [ ] Installed `black` (i.e., `python -m pip install black` or `conda install -c conda-forge black`)
- [ ] Installed `flake8` (i.e., `python -m pip install flake8` or `conda install -c conda-forge flake8`)
- [ ] Installed `pytest-cov` (i.e., `python -m pip install pytest-cov` or `conda install -c conda-forge pytest-cov`)
- [ ] Ran `black --exclude=".*\.ipynb" --extend-exclude=".venv" --diff ./` in the root stumpy directory
- [ ] Ran `flake8 --extend-exclude=.venv ./` in the root stumpy directory
- [ ] Ran `./setup.sh dev && ./test.sh` in the root stumpy directory
