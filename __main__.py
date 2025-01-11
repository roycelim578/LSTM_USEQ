#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:05:16 2024

@author: roycelim
"""

##############################################
# Libraries
##############################################

import os
import concurrent.futures
import subprocess
from pathlib import Path


##############################################
# Set Up
##############################################

# Directories
wd = str(Path(__file__).resolve().parents[0])

lib_phases = [
    'DATA_PROCESSING',
    'MODEL_TRAINING',
    'STRATEGY_BACKTESTING',
    'FINAL_SUMMARY'
]

# Settings
phases = [2, 3]  # List of phase indices to run; modify as needed

##############################################
# Execution
##############################################

# Functions
def execute_script(script_path):
    """
    Function to execute a single script.
    """
    try:
        print(f"Executing {script_path}")
        subprocess.check_call(['python3', script_path])
        print(f"Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}. Error: {e}")

# Run Scripts
if __name__ == "__main__":
    for i in phases:

        phase = lib_phases[i]
        print(f"\nStarting phase: {phase}")

        # Construct the path to the scripts directory for the current phase
        scripts_dir = os.path.join(wd, 'SCRIPTS', phase)
        print(f"Scripts directory: {scripts_dir}")

        # Check if the scripts directory exists
        if not os.path.isdir(scripts_dir):
            print(f"Scripts directory '{scripts_dir}' does not exist. Skipping phase '{phase}'.")
            continue

        # List all Python scripts in the scripts directory
        try:
            scripts = [
                os.path.join(scripts_dir, script)
                for script in os.listdir(scripts_dir)
                if script.endswith('.py') and os.path.isfile(os.path.join(scripts_dir, script))
            ]
            if not scripts:
                print(f"No Python scripts found in '{scripts_dir}'. Skipping phase '{phase}'.")
                continue
            print(f"Scripts to execute: {scripts}")
        except Exception as e:
            print(f"An error occurred while listing scripts in '{scripts_dir}': {e}")
            continue

        # Execute all scripts concurrently
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(scripts)) as executor:
            # Submit all scripts to the executor
            futures = {executor.submit(execute_script, script): script for script in scripts}

            # Monitor the execution
            for future in concurrent.futures.as_completed(futures):
                script = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Unhandled exception occurred while executing {script}: {e}")

        print(f"All scripts completed for phase: {phase}\n")
