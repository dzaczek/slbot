import numpy as np
import pytest
from per import SumTree

def test_sum_tree_add_and_total():
    capacity = 10
    tree = SumTree(capacity)

    # Add items
    for i in range(capacity):
        tree.add(1.0, f"data_{i}")

    assert tree.total() == float(capacity)
    assert tree.count == capacity

    # Check if tree property holds (root sum = sum of all leaves)
    # Leaves are at indices capacity-1 to 2*capacity-2
    leaves_sum = np.sum(tree.tree[capacity-1:])
    assert tree.total() == leaves_sum

def test_sum_tree_update():
    capacity = 10
    tree = SumTree(capacity)

    for i in range(capacity):
        tree.add(1.0, f"data_{i}")

    # Update first item (which was added at idx = capacity - 1 + 0)
    idx = capacity - 1
    new_p = 5.0
    tree.update(idx, new_p)

    assert tree.total() == (capacity - 1) + new_p
    assert tree.tree[idx] == new_p

def test_sum_tree_retrieve():
    capacity = 4
    tree = SumTree(capacity)

    # Tree structure for capacity=4:
    # Size = 7.
    # Indices: 0 (root), 1, 2 (level 1), 3, 4, 5, 6 (leaves)
    # Leaves are 3, 4, 5, 6.

    tree.add(10.0, "A") # idx 3
    tree.add(20.0, "B") # idx 4
    tree.add(30.0, "C") # idx 5
    tree.add(40.0, "D") # idx 6

    assert tree.total() == 100.0

    # Retrieve logic:
    # s in [0, 10) -> idx 3
    # s in [10, 30) -> idx 4
    # s in [30, 60) -> idx 5
    # s in [60, 100) -> idx 6

    idx, p, data = tree.get(5.0)
    assert idx == 3
    assert data == "A"

    idx, p, data = tree.get(15.0)
    assert idx == 4
    assert data == "B"

    idx, p, data = tree.get(45.0)
    assert idx == 5
    assert data == "C"

    idx, p, data = tree.get(80.0)
    assert idx == 6
    assert data == "D"

def test_sum_tree_retrieve_boundary():
    capacity = 4
    tree = SumTree(capacity)
    tree.add(10.0, "A")
    tree.add(20.0, "B")
    tree.add(30.0, "C")
    tree.add(40.0, "D")

    # Exactly on boundary
    # s=10 -> current implementation uses <=, so it goes to left child if s == left_val
    # left child of 1 is 3 (val 10). s=10 <= 10. Go to 3.
    idx, p, data = tree.get(10.0)
    assert idx == 3

    # s=0 -> idx 3
    idx, p, data = tree.get(0.0)
    assert idx == 3

def test_sum_tree_overwrite():
    capacity = 2
    tree = SumTree(capacity)

    tree.add(1.0, "A")
    tree.add(1.0, "B")
    assert tree.total() == 2.0

    # Overwrite A
    tree.add(2.0, "C")
    # Buffer is circular. Write pointer wraps around.
    # A was at idx 1 (capacity-1=1). Write=0.
    # B was at idx 2. Write=1.
    # C should overwrite A at idx 1.

    assert tree.total() == 3.0 # 1 (B) + 2 (C)
    idx, p, data = tree.get(1.0) # s=1.0. First leaf (C) covers [0, 2). B covers [2, 3).
    # Wait, order of leaves:
    # Leaves are at indices capacity-1 to 2*capacity-2.
    # For capacity=2: leaves at 1, 2.
    # Write=0 -> writes to idx 1.
    # Write=1 -> writes to idx 2.
    # Step 1: add A. write=0 -> idx 1. tree[1]=1. write=1.
    # Step 2: add B. write=1 -> idx 2. tree[2]=1. write=0.
    # Step 3: add C. write=0 -> idx 1. tree[1]=2. write=1.

    # Tree:
    # 0 -> 3.0
    # 1 (left) -> 2.0 (C)
    # 2 (right) -> 1.0 (B)

    # get(1.0):
    # s=1.0 <= tree[1] (2.0) -> go left -> 1.
    assert idx == 1
    assert data == "C"
