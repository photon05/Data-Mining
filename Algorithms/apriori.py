from collections import defaultdict
from itertools import combinations

def load_dataset():
    # Sample dataset
    dataset = [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]
    return dataset

def create_candidates(dataset):
    # Create the list of unique items in the dataset
    items = set()
    for transaction in dataset:
        for item in transaction:
            items.add(frozenset([item]))

    return items

def filter_candidates(dataset, candidates, min_support):
    # Count the occurrences of candidates in the dataset
    candidate_counts = defaultdict(int)
    for transaction in dataset:
        for candidate in candidates:
            if candidate.issubset(transaction):
                candidate_counts[candidate] += 1

    # Filter candidates that meet the minimum support threshold
    num_transactions = float(len(dataset))
    frequent_candidates = {}
    for candidate, count in candidate_counts.items():
        support = count / num_transactions
        if support >= min_support:
            frequent_candidates[candidate] = support

    return frequent_candidates

def generate_candidates(prev_candidates, k):
    # Generate new candidates of size k from the previous frequent item sets
    new_candidates = set()
    for item1 in prev_candidates:
        for item2 in prev_candidates:
            union_candidate = item1.union(item2)
            if len(union_candidate) == k:
                new_candidates.add(union_candidate)

    return new_candidates

def apriori(dataset, min_support):
    frequent_itemsets = {}  # Dictionary to store frequent item sets
    candidates = create_candidates(dataset)
    k = 1
    while candidates:
        frequent_candidates = filter_candidates(dataset, candidates, min_support)
        frequent_itemsets[k] = frequent_candidates
        k += 1
        candidates = generate_candidates(frequent_candidates, k)

    return frequent_itemsets

if __name__ == "__main__":
    dataset = load_dataset()
    min_support = 0.5
    frequent_itemsets = apriori(dataset, min_support)
    print("Frequent Itemsets:")
    for k, itemsets in frequent_itemsets.items():
        print(f"k = {k}")
        for itemset, support in itemsets.items():
            print(f"{itemset}: {support}")
