#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

uv run --with-requirements $SCRIPT_DIR/requirements.txt $SCRIPT_DIR/run.py "$@"
