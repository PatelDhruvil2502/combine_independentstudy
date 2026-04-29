import os
import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    print("Capturing Real Memory Addresses...")
    
    num_elements = 1000
    dim = 128
    
    # --- 1. Sequential Storage Layout (NumPy Array) ---
    # Creates a single, contiguous block of memory for 1,000 vectors
    array = np.zeros((num_elements, dim), dtype=np.float32)
    
    # Get the raw base memory address of the C-array inside NumPy
    base_address = array.__array_interface__['data'][0]
    stride = array.strides[0] # Exact number of bytes to jump to the next row (128 floats * 4 bytes = 512 bytes)
    
    # Extract exact physical memory addresses
    seq_addresses = [base_address + i * stride for i in range(num_elements)]
    
    
    # --- 2. Random/Dynamic Storage Layout (Graph / Heap Objects) ---
    # Simulates how a dynamic graph like HNSW allocates nodes over time.
    class GraphNode:
        def __init__(self, node_id):
            self.id = node_id
            self.vector = [0.0] * 128

    nodes = []
    # Real-world heaps get fragmented. We allocate lists randomly to mimic 
    # other parts of the program claiming memory in between our graph nodes.
    junk = []
    for i in range(num_elements):
        nodes.append(GraphNode(i))
        # Random background allocations that fragment the standard Python heap
        for _ in range(random.randint(0, 5)):
            junk.append([0.0] * random.randint(10, 50))
            
    # In CPython, id(object) returns the exact physical RAM address
    graph_addresses = [id(node) for node in nodes]
    
    
    # --- Normalize Addresses to KB offset (So we can compare them visually) ---
    # We subtract the lowest address from each dataset so they start at 0
    seq_min = min(seq_addresses)
    norm_seq_kb = [(addr - seq_min) / 1024 for addr in seq_addresses]
    
    graph_min = min(graph_addresses)
    norm_graph_kb = [(addr - graph_min) / 1024 for addr in graph_addresses]
    
    
    # --- Plotting 1: How Data is Stored ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_title("Physical Memory Layout (How it's Stored)", fontweight='bold')
    ax1.scatter(range(num_elements), norm_seq_kb, color='blue', alpha=0.6, s=10, label='NumPy Array (Contiguous)')
    ax1.scatter(range(num_elements), norm_graph_kb, color='red', alpha=0.4, s=10, label='Graph Nodes (Fragmented Heap)')
    ax1.set_xlabel("Element / Node Index")
    ax1.set_ylabel("Memory Address Offset (KB)")
    ax1.legend()

    
    # --- Plotting 2: How Data is Accessed (The Latency Cause) ---
    ax2.set_title("Memory Access Pattern (How it's Read)", fontweight='bold')
    
    # Simulate a Sequential Read (NumPy Scan)
    seq_read_addrs = [norm_seq_kb[i] for i in range(100)]
    ax2.plot(range(100), seq_read_addrs, 'bo-', alpha=0.8, markersize=4, label='Sequential Scan')
    
    # Simulate a Graph Traversal (Jumping to random neighbors)
    rand_read_addrs = []
    curr = 0
    for _ in range(100):
        rand_read_addrs.append(norm_graph_kb[curr])
        curr = random.randint(0, num_elements - 1)
        
    ax2.plot(range(100), rand_read_addrs, 'ro-', alpha=0.8, markersize=4, lw=1.5, label='Graph Traversal (Random Jumps)')
    
    ax2.set_xlabel("Query / Step Number")
    ax2.set_ylabel("Memory Address Accessed (KB)")
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = "results/memory_layout_visualization.png"
    plt.savefig(out_path, dpi=300)
    print(f"Graph saved: {out_path}")

if __name__ == "__main__":
    main()
