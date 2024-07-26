# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from collections import defaultdict


def main(args):
    with open(os.path.join(args.output_dir, "merge.json")) as file:
        ans_file = file.readlines()
    data = [json.loads(line) for line in ans_file]

    match_cnt, total_cnt = 0, 0
    category_counts = defaultdict(lambda: {"match": 0, "total": 0})

    for item in data:
        pred, answer = item["pred"], item["answer"]
        if len(pred) > 1:
            pred = pred[0]
        match_cnt += pred == answer
        total_cnt += 1

        # Get the category and update category-specific counts
        category_key = item["question_category"]
        category_counts[category_key]["match"] += pred == answer
        category_counts[category_key]["total"] += 1

    accs = []
    # Print overall accuracy
    print(f"Total: {total_cnt}, Correct: {match_cnt}, Accuracy: {match_cnt/total_cnt*100:.2f}%")
    accs.append(match_cnt / total_cnt * 100)

    # Print accuracy for each category
    for category in ["CRD", "NPA", "STA", "TEMP", "TH"]:
        match = category_counts[category]["match"]
        total = category_counts[category]["total"]
        accuracy = (match / total * 100) if total > 0 else 0
        print(f"Category: {category}, Total: {total}, Correct: {match}, Accuracy: {accuracy:.2f}%")
        accs.append(accuracy)

    # Print the accuracies as a comma-separated list for spreadsheet logging
    print(",".join([str(acc) for acc in accs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
