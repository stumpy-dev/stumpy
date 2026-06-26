#!/usr/bin/env python

import argparse
import json
import re
from urllib import request

import pandas as pd
from docutils import nodes
from docutils.core import publish_doctree
from packaging.specifiers import SpecifierSet
from packaging.version import Version

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
}


def get_all_python_versions():
    """
    Retrieve all supported Python versions (i.e., not end-of-life)

    Note that all versions are returned in descending order
    """
    python_versions = (
        pd.read_html(
            "https://devguide.python.org/versions/",
            storage_options=HEADERS,
        )[0]
        .query('Branch != "main"')
        .dropna()
        .Branch.to_list()
    )
    return python_versions


def get_all_python_versions_in_range(start, stop):
    """
    Find all Python minor versions within some start/stop range (inclusive)
    """
    python_versions = get_all_python_versions()[::-1]  # Reverse order
    start_idx = python_versions.index(start)
    stop_idx = python_versions.index(stop)
    print(json.dumps(python_versions[start_idx : stop_idx + 1]))


def get_safe_python_version():
    """
    Retrieve the n-2 Python release version
    """
    python_versions = get_all_python_versions()[::-1]  # Reverse order
    print(python_versions[-3])


def get_min_python_version():
    """
    Find the minimum version of Python supported (i.e., not end-of-life)
    """
    min_python = get_all_python_versions()[-1]
    return min_python


def get_min_numba_numpy_version(min_python):
    """
    Find the minimum versions of Numba and NumPy that supports the specified
    `min_python` version
    """
    df = (
        pd.read_html(
            "https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information",  # noqa
            storage_options=HEADERS,
        )[0]
        .dropna()
        .drop(columns=["Numba.1", "llvmlite", "LLVM", "TBB"])
        .query('`Python`.str.contains("2.7") == False')
        .query('`Numba`.str.contains(".x") == False')
        .query('`Numba`.str.contains("{") == False')
        .pipe(
            lambda df: df.assign(
                MIN_PYTHON_SPEC=(
                    df.Python.str.split().str[1].replace({"<": "="}, regex=True)
                    + df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                ).apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MIN_NUMPY=(df.NumPy.str.split().str[0].replace({".x": ""}, regex=True))
            )
        )
        .assign(
            COMPATIBLE=lambda row: row.apply(
                check_python_compatibility, axis=1, args=(Version(min_python),)
            )
        )
        .query("COMPATIBLE == True")
        .pipe(lambda df: df.assign(MINOR=df.Numba.str.split(".").str[1]))
        .pipe(lambda df: df.assign(PATCH=df.Numba.str.split(".").str[2]))
        .sort_values(["MINOR", "PATCH"], ascending=[False, True])
        .iloc[-1]
    )
    return df.Numba, df.MIN_NUMPY


def check_python_compatibility(row, min_python):
    """
    Determine the Python version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC)
    return python_compatible


def check_scipy_compatibility(row, min_python, min_numpy):
    """
    Determine the Python and NumPy version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC & row.MAX_PYTHON_SPEC)
    numpy_compatible = min_numpy in (row.MIN_NUMPY_SPEC & row.MAX_NUMPY_SPEC)
    return python_compatible & numpy_compatible


def get_scipy_version_df_backup():
    """
    Retrieve raw SciPy version table as DataFrame
    """
    colnames = pd.read_html(
        "https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy",
        storage_options=HEADERS,
    )[1].columns
    converter = {colname: str for colname in colnames}
    return (
        pd.read_html(
            "https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy",
            storage_options=HEADERS,
            converters=converter,
        )[1]
        .rename(columns=lambda x: x.replace(" ", "_"))
        .replace({".x": ""}, regex=True)
    )


def get_scipy_version_df():
    """
    Retrieve raw SciPy version table as DataFrame

    This function retrieves the table directly from the SciPy Github docs and
    replaces `get_scipy_version_df_backup`.
    """
    url = (
        "https://raw.githubusercontent.com/scipy/scipy/"
        "refs/heads/main/doc/source/dev/toolchain.rst"
    )
    rst = request.urlopen(url).read()

    doctree = publish_doctree(rst, settings_overrides={"report_level": 5})

    extracted_table = []
    for node in doctree.findall(nodes.literal_block):
        if "Python and NumPy version support per SciPy version" in node.astext():
            idx = -1
            for i, l in enumerate(node.astext().split("\n")):
                if l.lstrip().startswith("="):
                    idx = i
                    break
            table_str = "\n".join(node.astext().split("\n")[idx:])
            table_node = publish_doctree(table_str)

            for row_node in table_node.findall(nodes.row):
                current_row = []
                for entry_node in row_node.findall(nodes.entry):
                    # Extract text from the cell
                    current_row.append(entry_node.astext())
                extracted_table.append(current_row)

    df = pd.DataFrame(extracted_table[1:])
    df.columns = extracted_table[0]

    return df.rename(columns=lambda x: x.replace(" ", "_"))


def get_min_scipy_version(min_python, min_numpy):
    """
    Determine the SciPy version compatibility
    """
    df = (
        get_scipy_version_df()
        .pipe(
            lambda df: df.assign(
                SciPy_version=df.SciPy_version.str.replace(
                    r"\d\/", "", regex=True  # noqa
                )
            )
        )
        .query('`Python_versions`.str.contains("2.7") == False')
        .pipe(
            lambda df: df.assign(
                MIN_PYTHON_SPEC=df.Python_versions.str.split(",")
                .str[0]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MAX_PYTHON_SPEC=df.Python_versions.str.split(",")
                .str[1]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MIN_NUMPY_SPEC=df.NumPy_versions.str.split(",")
                .str[0]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MAX_NUMPY_SPEC=df.NumPy_versions.str.split(",")
                .str[1]
                .apply(SpecifierSet)
            )
        )
        .assign(
            COMPATIBLE=lambda row: row.apply(
                check_scipy_compatibility,
                axis=1,
                args=(Version(min_python), Version(min_numpy)),
            )
        )
        .query("COMPATIBLE == True")
        .pipe(lambda df: df.assign(MINOR=df.SciPy_version.str.split(".").str[1]))
        .pipe(lambda df: df.assign(PATCH=df.SciPy_version.str.split(".").str[2]))
        .sort_values(["MINOR", "PATCH"], ascending=[False, True])
        .iloc[-1]
    )
    return df.SciPy_version


def get_minor_versions_between(start_version_str, end_version_str):
    """
    Returns a list of all minor Python versions between two specified minor versions
    """
    try:
        start_parts = [int(x) for x in start_version_str.split(".")]
        end_parts = [int(x) for x in end_version_str.split(".")]
    except ValueError:
        raise ValueError("Invalid version string format. Expected 'MAJOR.MINOR.PATCH'.")

    if len(start_parts) < 2 or len(end_parts) < 2:
        raise ValueError("Version string must include at least major and minor parts.")

    start_major, start_minor = start_parts[0], start_parts[1]
    end_major, end_minor = end_parts[0], end_parts[1]

    if start_major != end_major:
        print("Warning: Major versions differ. Returning an empty list.")
        return []

    if start_minor >= end_minor:
        print(
            "Warning: Start minor version is not less than end minor version."
            "Returning an empty list."
        )
        return []

    versions = []
    for minor in range(start_minor, end_minor + 1):
        versions.append(f"{start_major}.{minor}")

    return versions


def get_latest_scipy_version():
    """
    Reteive the latest SciPy version
    """
    return _get_all_pkg_versions("scipy")[-1]


def get_latest_scipy_python_version():
    """
    Retrieve the latest Python version that is supported by SciPy
    """
    python_version = get_latest_pkg_python_version("scipy")
    return f"<={python_version}"


def get_latest_pkg_python_version(pkg):
    """
    Retrieve the latest Python version that is supported by <pkg>
    """
    python_versions = []
    response = _get_pypi_json(pkg)
    for classifier in response.get("info").get("classifiers"):
        if "Python" in classifier:
            match = re.search(r"Python\s+\:\:\s+(\d+\.\d+)", classifier)
            if match is not None:
                python_versions.append(match.groups()[0])
    python_version = sorted(python_versions, key=Version)[-1]
    return python_version


def _get_pypi_json(pkg):
    url = f"https://pypi.python.org/pypi/{pkg}/json"
    response = json.loads(request.urlopen(url).read())
    return response


def _get_all_pkg_versions(pkg):
    response = _get_pypi_json(pkg)
    all_versions = response["releases"]
    all_versions = [version for version in all_versions.keys() if "rc" not in version]
    all_versions = sorted(all_versions, key=Version)
    return all_versions


def get_latest_numpy_versions(n=10):
    """
    Retrieve the n latest NumPy versions
    """
    return _get_all_pkg_versions("numpy")[-n:]


def check_python_version(row):
    """
    Ensure that the Python version is compatible with Numba and SciPy
    """
    versions = get_minor_versions_between(
        row.START_PYTHON_VERSION, row.END_PYTHON_VERSION
    )

    compatible_version = None
    for version in versions:
        if Version(version) in row.NUMBA_PYTHON_SPEC & row.SCIPY_PYTHON_SPEC:
            compatible_version = version
    return compatible_version


def check_numpy_version(row):
    """
    Ensure that the NumPy version is compatible with the NumPy Specs
    """
    out = None
    for numpy_version in row.NUMPY:
        if numpy_version in row.NUMPY_SPEC:
            out = numpy_version
    return out


def get_all_max_versions():
    """
    Find the maximum version of Python that is compatible with Numba and NumPy
    """
    df = (
        pd.read_html(
            "https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information",  # noqa
            storage_options=HEADERS,
        )[0]
        .dropna()
        .drop(columns=["Numba.1", "llvmlite", "LLVM", "TBB"])
        .query('`Python`.str.contains("2.7") == False')
        .query('`Numba`.str.contains(".x") == False')
        .query('`Numba`.str.contains("{") == False')
        .pipe(
            lambda df: df.assign(
                START_PYTHON_VERSION=(
                    df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                )
            )
        )
        .pipe(
            lambda df: df.assign(
                END_PYTHON_VERSION=(
                    df.Python.str.split().str[4].replace({".x": ""}, regex=True)
                )
            )
        )
        .pipe(
            lambda df: df.assign(
                NUMBA_PYTHON_SPEC=(
                    df.Python.str.split().str[1].replace({"<": ">"}, regex=True)
                    + df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                    + ", "
                    + df.Python.str.split().str[3]
                    + df.Python.str.split().str[4].replace({".x": ""}, regex=True)
                ).apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                NUMPY=([get_latest_numpy_versions() for i in df.index])
            )
        )
        .pipe(
            lambda df: df.assign(
                NUMPY_SPEC=(
                    df.NumPy.str.replace(r" [;†\.]$", "", regex=True)
                    .str.split()
                    .str[-2:]
                    .replace({".x": ""}, regex=True)
                    .str.join("")
                ).apply(SpecifierSet)
            )
        )
        .assign(NUMPY=lambda row: row.apply(check_numpy_version, axis=1))
        .assign(SCIPY=get_latest_scipy_version())
        .assign(SCIPY_PYTHON_SPEC=get_latest_scipy_python_version())
        .pipe(
            lambda df: df.assign(
                SCIPY_PYTHON_SPEC=df.SCIPY_PYTHON_SPEC.apply(SpecifierSet)
            )
        )
        .assign(MAX_PYTHON=lambda row: row.apply(check_python_version, axis=1))
        .pipe(lambda df: df.assign(MAJOR=df.MAX_PYTHON.str.split(".").str[0]))
        .pipe(lambda df: df.assign(MINOR=df.MAX_PYTHON.str.split(".").str[1]))
        .sort_values(["MAJOR", "MINOR"], ascending=[False, False])
        .iloc[0]
    )

    print(
        f"python: {df.MAX_PYTHON}\n"
        f"numba: {df.Numba}\n"
        f"numpy: {df.NUMPY}\n"
        f"scipy: {df.SCIPY}"
    )


def match_pkg_version(line, pkg_name):
    """
    Regular expression to match package versions
    """
    matches = re.search(
        rf"""
                        {pkg_name}  # Package name
                        [\s=><:"\'\[\]]*  # Zero or more spaces or special characters
                        (\d+\.\d+[\.0-9]*)  # Capture "version" in `matches`
                        """,
        line,
        re.VERBOSE | re.IGNORECASE,  # Ignores all whitespace and case in pattern
    )

    return matches


def find_pkg_mismatches(pkg_name, pkg_version, fnames):
    """
    Determine if any package version has mismatches
    """
    pkg_mismatches = []

    for fname in fnames:
        with open(fname, "r") as file:
            for line_num, line in enumerate(file, start=1):
                l = line.strip().replace(" ", "").lower()
                matches = match_pkg_version(l, pkg_name)
                if matches is not None:
                    version = matches.groups()[0]
                    if version != pkg_version:
                        pkg_mismatches.append((pkg_name, version, fname, line_num))

    return pkg_mismatches


def test_pkg_mismatch_regex():
    """
    Validation function for the package mismatch regex
    """
    pkgs = {
        "numpy": "0.0",
        "scipy": "0.0",
        "python": "2.7",
        "python-version": "2.7",
        "numba": "0.0",
    }

    lines = [
        "Programming Language :: Python :: 3.8",
        "STUMPY supports Python 3.8",
        "python-version: ['3.8']",
        'requires-python = ">=3.8"',
        "numba>=0.55.2",
    ]

    for line in lines:
        match_found = False
        for pkg_name, pkg_version in pkgs.items():
            matches = match_pkg_version(line, pkg_name)

            if matches:
                match_found = True
                break

        if not match_found:
            raise ValueError(f'Package mismatch regex fails to cover/match "{line}"')


def get_all_min_versions(MIN_PYTHON):
    """
    Retrieve all minimum Python and package versions
    """
    MIN_NUMBA, MIN_NUMPY = get_min_numba_numpy_version(MIN_PYTHON)
    MIN_SCIPY = get_min_scipy_version(MIN_PYTHON, MIN_NUMPY)

    print(
        f"python: {MIN_PYTHON}\n"
        f"numba: {MIN_NUMBA}\n"
        f"numpy: {MIN_NUMPY}\n"
        f"scipy: {MIN_SCIPY}"
    )

    pkgs = {
        "numpy": MIN_NUMPY,
        "scipy": MIN_SCIPY,
        "numba": MIN_NUMBA,
        "python": MIN_PYTHON,
        "python-version": MIN_PYTHON,
    }

    fnames = [
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        ".github/workflows/github-actions.yml",
        "README.rst",
    ]

    with open("pyproject.toml", "r") as f:
        pyproject_lines = f.readlines()

    test_pkg_mismatch_regex()

    for pkg_name, pkg_version in pkgs.items():
        for name, version, fname, line_num in find_pkg_mismatches(
            pkg_name, pkg_version, fnames
        ):
            if fname == "pyproject.toml":
                line = pyproject_lines[line_num - 1]
                if "Programming Language :: Python" in line and Version(
                    pkg_version
                ) <= Version(version):
                    # Skip lines in `pyproject.toml` where the "Programming Language"
                    # version may be higher than the minimum Python version
                    continue

            print(
                f"{pkg_name} {pkg_version} Mismatch: Version {version} "
                f"found in {fname}:{line_num}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mode",
        type=str,
        default="min",
        help='Options: ["min", "max", "range", "safe", "latest"]',
    )
    parser.add_argument(
        "-pkg", type=str, default=None, help="Name of any Python package in PyPI"
    )
    parser.add_argument("python_version", nargs="*", default=None)
    args = parser.parse_args()
    # Example
    # ./versions.py
    # ./versions.py -pkg ray
    # ./versions.py -mode min
    # ./versions.py -mode max
    # ./versions.py -mode safe
    # ./versions.py -mode latest
    # /versions.py -mode range 3.10 3.14

    if args.pkg:
        print(get_latest_pkg_python_version(args.pkg))
    elif args.mode == "min":
        if len(args.python_version) == 0:
            args.python_version = None
        elif len(args.python_version) == 1:
            args.python_version = args.python_version[0]

        if args.python_version is not None:
            MIN_PYTHON = str(args.python_version)
        else:
            MIN_PYTHON = get_min_python_version()
        get_all_min_versions(MIN_PYTHON)
    elif args.mode == "max":
        get_all_max_versions()
    elif args.mode == "range":
        if len(args.python_version) != 2:
            raise IndexError("Please provide both a start and stop version")
        start = args.python_version[0]
        stop = args.python_version[1]
        get_all_python_versions_in_range(start, stop)
    elif args.mode == "safe":
        get_safe_python_version()
    elif args.mode == "latest":
        print(get_all_python_versions()[0])
    else:
        raise ValueError(f'Unrecognized mode: "{args.mode}"')
