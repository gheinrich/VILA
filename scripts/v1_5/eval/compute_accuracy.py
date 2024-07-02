import argparse, os
import json
from collections import defaultdict


def main(args):
    with open(os.path.join(args.output_dir, 'merge.json'), 'r') as file:
        ans_file = file.readlines()
    data = [json.loads(line) for line in ans_file]

    match_cnt, total_cnt = 0, 0
    category_counts = defaultdict(lambda: {'match': 0, 'total': 0})
    
    for item in data:
        pred, answer = item['pred'], item['answer']
        if len(pred) > 1:
            pred = pred[0]
        match_cnt += pred == answer
        total_cnt += 1

        # Get the category and update category-specific counts
        category_key = item['question_category']
        category_counts[category_key]['match'] += pred == answer
        category_counts[category_key]['total'] += 1

    accs = []
    # Print overall accuracy
    print(f"Total: {total_cnt}, Correct: {match_cnt}, Accuracy: {match_cnt/total_cnt*100:.2f}%")
    accs.append(match_cnt/total_cnt*100)

    # Print accuracy for each category
    for category in ['CRD', 'NPA', 'STA', 'TEMP', 'TH']:
        match = category_counts[category]['match']
        total = category_counts[category]['total']
        accuracy = (match / total * 100) if total > 0 else 0
        print(f"Category: {category}, Total: {total}, Correct: {match}, Accuracy: {accuracy:.2f}%")
        accs.append(accuracy)
    
    # Print the accuracies as a comma-separated list for spreadsheet logging
    print(','.join([str(acc) for acc in accs]))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)