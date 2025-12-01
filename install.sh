#!/usr/bin/env bash

# Install the octi.lab, octi.lab_assets, and octi.lab_tasks packages in editable mode

# Exit if any command fails
set -e

# Base path to the extensions
BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/extensions"

#
echo "Installing all packages from ${BASE_PATH}..."

# Install each package in editable mode
echo "Installing diff.lab..."
pip install -e "${BASE_PATH}/diff.lab"

echo "Installing diff.lab_assets..."
pip install -e "${BASE_PATH}/diff.lab_assets"

echo "Installing diff.lab_tasks..."
pip install -e "${BASE_PATH}/diff.lab_tasks[all]"

# echo "Installing diff.lab_apps..."
# pip install -e "${BASE_PATH}/diff.lab_apps"

echo "All packages have been installed in editable mode."