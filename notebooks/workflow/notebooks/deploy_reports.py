#!/bin/python3 

import re
import os, os.path
import markdown
from pathlib import Path

SOURCE_DIR = Path(".")
TARGET_DIR = Path("../docs")
GITHUB_BRANCH = "https://github.com/landerlini/lb-trksim-train/tree/notebooks/notebooks"

dest_from_source = {
        "index.html": "README.md",
        "deploy.html": "reports/deploy.html",
        "preprocessing.html": "reports/preprocessing.html",
        "preprocessing_gans.html": "reports/preprocessing_gans.html",
        "train_acceptance.html": "reports/train_acceptance.html",
        "train_covariance.html": "reports/train_covariance.html",
        "train_resolution.html": "reports/train_resolution.html",
        "train_efficiency.html": "reports/train_efficiency.html",
        "validate_acceptance.html": "reports/validate_acceptance.html",
        "validate_covariance.html": "reports/validate_covariance.html",
        "validate_resolution.html": "reports/validate_resolution.html",
        "validate_efficiency.html": "reports/validate_efficiency.html",
        }

links = {
        "./Preprocessing.ipynb": "./preprocessing.html",
        "./Preprocessing-GANs.ipynb": "./preprocessing_gans.html",
        "./Acceptance.ipynb": "./train_acceptance.html",
        "./Efficiency.ipynb": "./train_efficiency.html",
        "./Resolution.ipynb": "./train_resolution.html",
        "./Covariance.ipynb": "./train_covariance.html",
        "./Acceptance-validation.ipynb": "./validate_acceptance.html",
        "./Efficiency-validation.ipynb": "./validate_efficiency.html",
        "./Resolution-validation.ipynb": "./validate_resolution.html",
        "./Covariance-validation.ipynb": "./validate_covariance.html",
        "./Deploy.ipynb": "./deploy.html",
        }


def main():
    for dest_, source_ in dest_from_source.items():
        dest = TARGET_DIR/dest_
        source = SOURCE_DIR/source_

        # Clean the target
        try: 
            os.remove(dest)
        except FileNotFoundError:
            pass 

        if source_.lower().endswith(".md"):
            convert_from_md (source, dest)
        elif source_.lower().endswith(".html") or source_.lower().endswith(".htm"):
            convert_from_html (source, dest)
        else:
            print (f"Warning, unexpected source file {source_} is copy-pasted")
            os.system(f"cp {source} {dest}")

def convert_from_md(source, dest):
    with open(dest, "w") as outfile:
        with open(source) as infile:
            outfile.write(fix_links(markdown.markdown(infile.read())))


def convert_from_html(source, dest):
    with open(dest, "w") as outfile:
        with open(source) as infile:
            html = infile.read()
            html = clean_stderr(html)
            html = fix_links(html)
            outfile.write(html)


def clean_stderr(html):
    return re.sub(r"<div [^>]+vnd\.jupyter\.stderr[^>]*>", "<div style='visibility: hidden;'>", html)

def fix_links(html):
    for link_old, link_new in links.items():
        html = html.replace(f"href=\"{link_old}\"", f"href=\"{link_new}\"")

    html = re.sub(r'href="\./([^"]*\.py)"', f"href=\"{GITHUB_BRANCH}/\g<1>\"", html)
    html = re.sub(r'href="\./([^"]*\.c)"', f"href=\"{GITHUB_BRANCH}/\g<1>\"", html)
    html = re.sub(r'href="\./Snakefile"', f"href=\"{GITHUB_BRANCH}/Snakefile\"", html)

    return html

    


if __name__ == '__main__':
    main()
