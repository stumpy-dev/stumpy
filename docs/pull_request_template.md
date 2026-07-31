# Pull Request Checklist

Below is a simple checklist but please do not hesitate to ask for assistance!

- [ ] Read our [Contributing Guide](https://stumpy.readthedocs.io/en/latest/Contribute.html)
- [ ] Referenced a Github issue (or create one if one doesn't already exist)
- [ ] Read and reviewed all of the comments in the Github issue that you've referenced (along with cross-referenced issues/pull requests) to ensure that the issue still requires a pull request
- [ ] Checked that the issue has not already been assigned to anybody else or is already being addressed in another pull request
- [ ] Left a meaningful comment on the original Github issue to discuss the detailed approach for your contribution and received confirmation from the maintainers before proceeding with this pull request
- [ ] Forked, cloned, and checked out the newest version of the code
- [ ] Created a new branch
- [ ] Made necessary code changes
- [ ] Installed `black` (i.e., `python -m pip install black` or `conda install -c conda-forge black`)
- [ ] Installed `flake8` (i.e., `python -m pip install flake8` or `conda install -c conda-forge flake8`)
- [ ] Installed `pytest-cov` (i.e., `python -m pip install pytest-cov` or `conda install -c conda-forge pytest-cov`)
- [ ] Ran `black --exclude=".*\.ipynb" --extend-exclude=".venv" --diff ./` in the root stumpy directory
- [ ] Ran `flake8 --extend-exclude=.venv ./` in the root stumpy directory
- [ ] Ran `./setup.sh dev && ./test.sh` in the root stumpy directory and ensured that all tests are passing locally
- [ ] Check this box if AI code assistance was used to generate 15%+ of the code in this pull request

Please do not commit any code to avoid/circumvent a failing test and, instead, engage in a discussion (below) to determine the best course of action. 

Only request a review **after** the checklist above is fully completed!
