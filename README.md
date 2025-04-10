# AIS-NLP10S
AI School Course Excersises

https://github.com/astral-sh/uv

**STEP1**
Package Installations Instruction using 'uv':

Install uv with our standalone installers:

# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Or, from PyPI:

# With pip.
pip install uv

# Or pipx.
pipx install uv

*If installed via the standalone installer, uv can update itself to the latest version*

uv self update

*See the installation documentation for details and alternative installation methods.*

Documentation:
uv's documentation is available at docs.astral.sh/uv.

uv manages project dependencies and environments, with support for lockfiles, workspaces, and more, similar to rye or poetry:

**STEP2: Initialize Project**
goto desired folder, assume 'example'
$ cd example

Initialize project
$ uv init
Initialized project `example` at `/home/user/example`

**STEP3: Initialize environment**
$ uv venv
OR
$ uv venv --python 3.12.0
Using Python 3.12.0
Creating virtual environment at: .venv

**IMPORTANT step before installation packages**
Activate with: source .venv/bin/activate


**STEP3.1: Add required packages**
$ uv add pypdf [replace pypdf with desired package name]
Creating virtual environment at: .venv
Resolved 2 packages in 170ms.....

Just checking..
$ uv run pypdf check
All checks passed!


**STEP3.2: Optional steps**
$ uv lock
Resolved 2 packages in 0.33ms

Updates all packages at any time you run this
$ uv sync
Resolved 2 packages in 0.70ms
Audited 1 package in 0.02ms

Use a specific Python version in the current directory:
$ uv python pin 3.11
Pinned `.python-version` to `3.11`

$ uv pip compile requirements.txt \
   --universal \
   --output-file requirements.txt



**STEP4: Install packages using requirements.txt** 
$ uv pip sync requirements.txt
Resolved xx packages in 11ms
 ...

**WHENEVER YOU START WORK, YOU SWITCH THE ENVIRONNMENT**
1. Goto your project folder 
2. The activate with: 
$ source .venv/bin/activate

To exit out of environment:
$ deactivate

NOTE: Further help available at:
https://docs.astral.sh/uv/pip/environments/ 