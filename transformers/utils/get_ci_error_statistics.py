import argparse
import json
import math
import os
import subprocess
import time
import zipfile
from collections import Counter

import requests


def get_artifacts_links(worflow_run_id):
    """Get all artifact links from a workflow run"""

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{worflow_run_id}/artifacts?per_page=100"
    result = requests.get(url).json()
    artifacts = {}

    try:
        artifacts.update({artifact["name"]: artifact["archive_download_url"] for artifact in result["artifacts"]})
        pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)

        for i in range(pages_to_iterate_over):
            result = requests.get(url + f"&page={i + 2}").json()
            artifacts.update({artifact["name"]: artifact["archive_download_url"] for artifact in result["artifacts"]})

        return artifacts
    except Exception as e:
        print("Unknown error, could not fetch links.", e)

    return {}


def download_artifact(artifact_name, artifact_url, output_dir, token):
    """Download a GitHub Action artifact from a URL.

    The URL is of the from `https://api.github.com/repos/huggingface/transformers/actions/artifacts/{ARTIFACT_ID}/zip`,
    but it can't be used to download directly. We need to get a redirect URL first.
    See https://docs.github.com/en/rest/actions/artifacts#download-an-artifact
    """
    # Get the redirect URL first
    cmd = f'curl -v -H "Accept: application/vnd.github+json" -H "Authorization: token {token}" {artifact_url}'
    output = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o = output.stdout.decode("utf-8")
    lines = o.splitlines()

    for line in lines:
        if line.startswith("< Location: "):
            redirect_url = line[len("< Location: ") :]
            r = requests.get(redirect_url, allow_redirects=True)
            p = os.path.join(output_dir, f"{artifact_name}.zip")
            open(p, "wb").write(r.content)
            break


def get_errors_from_single_artifact(artifact_zip_path):
    """Extract errors from a downloaded artifact (in .zip format)"""
    errors = []
    failed_tests = []

    with zipfile.ZipFile(artifact_zip_path) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename):
                # read the file
                if filename in ["failures_line.txt", "summary_short.txt"]:
                    with z.open(filename) as f:
                        for line in f:
                            line = line.decode("UTF-8").strip()
                            if filename == "failures_line.txt":
                                try:
                                    # `error_line` is the place where `error` occurs
                                    error_line = line[: line.index(": ")]
                                    error = line[line.index(": ") + len(": ") :]
                                    errors.append([error_line, error])
                                except Exception:
                                    # skip un-related lines
                                    pass
                            elif filename == "summary_short.txt" and line.startswith("FAILED "):
                                # `test` is the test method that failed
                                test = line[len("FAILED ") :]
                                failed_tests.append(test)

    if len(errors) != len(failed_tests):
        raise ValueError(
            f"`errors` and `failed_tests` should have the same number of elements. Got {len(errors)} for `errors` "
            f"and {len(failed_tests)} for `failed_tests` instead. The test reports in {artifact_zip_path} have some"
            " problem."
        )

    return errors, failed_tests


def get_all_errors(artifact_dir):
    """Extract errors from all artifact files"""

    errors = []
    failed_tests = []

    paths = [os.path.join(artifact_dir, p) for p in os.listdir(artifact_dir) if p.endswith(".zip")]

    for p in paths:
        _errors, _failed_tests = get_errors_from_single_artifact(p)
        errors.extend(_errors)
        failed_tests.extend(_failed_tests)

    return errors, failed_tests


def reduce_by_error(logs, error_filter=None):
    """count each error"""

    counter = Counter()
    counter.update([x[1] for x in logs])
    counts = counter.most_common()
    r = {}
    for error, count in counts:
        if error_filter is None or error not in error_filter:
            r[error] = {"count": count, "failed_tests": [(x[2], x[0]) for x in logs if x[1] == error]}

    r = dict(sorted(r.items(), key=lambda item: item[1]["count"], reverse=True))
    return r


def get_model(test):
    """Get the model name from a test method"""
    test = test.split("::")[0]
    if test.startswith("tests/models/"):
        test = test.split("/")[2]
    else:
        test = None

    return test


def reduce_by_model(logs, error_filter=None):
    """count each error per model"""

    logs = [(x[0], x[1], get_model(x[2])) for x in logs]
    logs = [x for x in logs if x[2] is not None]
    tests = set([x[2] for x in logs])

    r = {}
    for test in tests:
        counter = Counter()
        # count by errors in `test`
        counter.update([x[1] for x in logs if x[2] == test])
        counts = counter.most_common()
        error_counts = {error: count for error, count in counts if (error_filter is None or error not in error_filter)}
        n_errors = sum(error_counts.values())
        if n_errors > 0:
            r[test] = {"count": n_errors, "errors": error_counts}

    r = dict(sorted(r.items(), key=lambda item: item[1]["count"], reverse=True))
    return r


def make_github_table(reduced_by_error):
    header = "| no. | error |"
    sep = "|-:|:-|"
    lines = [header, sep]
    for error in reduced_by_error:
        count = reduced_by_error[error]["count"]
        line = f"| {count} | {error[:100]} |"
        lines.append(line)

    return "\n".join(lines)


def make_github_table_per_model(reduced_by_model):
    header = "| model | no. of errors | major error | count |"
    sep = "|-:|-:|-:|-:|"
    lines = [header, sep]
    for model in reduced_by_model:
        count = reduced_by_model[model]["count"]
        error, _count = list(reduced_by_model[model]["errors"].items())[0]
        line = f"| {model} | {count} | {error[:60]} | {_count} |"
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--workflow_run_id", default=None, type=str, required=True, help="A GitHub Actions workflow run id."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Where to store the downloaded artifacts and other result files.",
    )
    parser.add_argument(
        "--token", default=None, type=str, required=True, help="A token that has actions:read permission."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    artifacts = get_artifacts_links(args.workflow_run_id)
    with open(os.path.join(args.output_dir, "artifacts.json"), "w", encoding="UTF-8") as fp:
        json.dump(artifacts, fp, ensure_ascii=False, indent=4)

    for idx, (name, url) in enumerate(artifacts.items()):
        download_artifact(name, url, args.output_dir, args.token)
        # Be gentle to GitHub
        time.sleep(1)

    errors, failed_tests = get_all_errors(args.output_dir)

    counter = Counter()
    counter.update([e[1] for e in errors])

    # print the top 30 most common test errors
    most_common = counter.most_common(30)
    for item in most_common:
        print(item)

    with open(os.path.join(args.output_dir, "errors.json"), "w", encoding="UTF-8") as fp:
        json.dump(errors, fp, ensure_ascii=False, indent=4)

    with open(os.path.join(args.output_dir, "failed_tests.json"), "w", encoding="UTF-8") as fp:
        json.dump(failed_tests, fp, ensure_ascii=False, indent=4)

    # Produce tables for GitHub issue.
    logs = [(error_line, error, failed_test) for (error_line, error), failed_test in zip(errors, failed_tests)]

    reduced_by_error = reduce_by_error(logs)
    reduced_by_model = reduce_by_model(logs)

    s1 = make_github_table(reduced_by_error)
    s2 = make_github_table_per_model(reduced_by_model)

    with open(os.path.join(args.output_dir, "reduced_by_error.txt"), "w", encoding="UTF-8") as fp:
        fp.write(s1)
    with open(os.path.join(args.output_dir, "reduced_by_model.txt"), "w", encoding="UTF-8") as fp:
        fp.write(s2)
