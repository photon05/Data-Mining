class TreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def increment(self, count):
        self.count += count

    def display(self, depth=0):
        print('  ' * depth, self.item, ' ', self.count)
        for child in self.children.values():
            child.display(depth + 1)

def load_dataset():
    # Sample dataset
    dataset = [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]
    return dataset

def create_item_counts(dataset):
    item_counts = {}
    for transaction in dataset:
        for item in transaction:
            if item not in item_counts:
                item_counts[item] = 1
            else:
                item_counts[item] += 1
    return item_counts

def build_tree(dataset, min_support):
    header_table = {}
    for transaction in dataset:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + 1

    for k in list(header_table.keys()):
        if header_table[k] < min_support:
            del(header_table[k])

    frequent_items = set(header_table.keys())
    if len(frequent_items) == 0:
        return None, None

    for item in header_table:
        header_table[item] = [header_table[item], None]

    fp_tree = TreeNode('Null', 1, None)

    for transaction in dataset:
        frequent_items_in_transaction = {}
        for item in transaction:
            if item in frequent_items:
                frequent_items_in_transaction[item] = header_table[item][0]

        if len(frequent_items_in_transaction) > 0:
            sorted_items = [v[0] for v in sorted(frequent_items_in_transaction.items(),
                                                  key=lambda p: p[1], reverse=True)]
            update_tree(sorted_items, fp_tree, header_table, 1)

    return fp_tree, header_table

def update_tree(items, tree_node, header_table, count):
    if items[0] in tree_node.children:
        tree_node.children[items[0]].increment(count)
    else:
        tree_node.children[items[0]] = TreeNode(items[0], count, tree_node)

        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = tree_node.children[items[0]]
        else:
            update_header(header_table[items[0]][1], tree_node.children[items[0]])

    if len(items) > 1:
        update_tree(items[1:], tree_node.children[items[0]], header_table, count)

def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node

def ascend_tree(node, prefix_path):
    if node.parent is not None:
        prefix_path.append(node.item)
        ascend_tree(node.parent, prefix_path)

def find_prefix_path(base_pat, tree_node):
    conditional_patterns = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            conditional_patterns[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return conditional_patterns

def mine_tree(tree_node, header_table, min_support, prefix, frequent_item_sets):
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]
    for base_pat in bigL:
        new_freq_set = prefix.copy()
        new_freq_set.add(base_pat)
        frequent_item_sets.append(new_freq_set)
        conditional_pattern_bases = find_prefix_path(base_pat, header_table[base_pat][1])
        conditional_tree, conditional_header_table = build_tree(conditional_pattern_bases, min_support)
        if conditional_header_table is not None:
            mine_tree(conditional_tree, conditional_header_table, min_support,
                      new_freq_set, frequent_item_sets)

def fp_growth(dataset, min_support):
    item_counts = create_item_counts(dataset)
    frequent_items = {item: count for item, count in item_counts.items() if count >= min_support}
    frequent_item_sets = []
    if len(frequent_items) == 0:
        return frequent_item_sets

    fp_tree, header_table = build_tree(dataset, min_support)
    if fp_tree is None:
        return frequent_item_sets

    mine_tree(fp_tree, header_table, min_support, set(), frequent_item_sets)
    return frequent_item_sets

if __name__ == "__main__":
    dataset = load_dataset()
    min_support = 2
    frequent_item_sets = fp_growth(dataset, min_support)
    print("Frequent Itemsets:")
    for itemset in frequent_item_sets:
        print(itemset)
