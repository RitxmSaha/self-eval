import json
import graphviz
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import os

def get_paths_to_node(root, target_node, current_path=None):
    """Find all paths from root to target node"""
    if current_path is None:
        current_path = []
    
    # Add current node's text to path
    current_path = current_path + [root.get('text', '')]
    
    # If this is our target node, return the path
    if root == target_node:
        return [current_path]
    
    # If no children or empty children, return empty list
    if not root.get('children'):
        return []
    
    # Recurse through children
    paths = []
    for child in root['children']:
        new_paths = get_paths_to_node(child, target_node, current_path)
        paths.extend(new_paths)
    
    return paths

def calculate_entropy(probabilities):
    """Calculate Shannon entropy from a dictionary of probabilities"""
    entropy = 0
    for p in probabilities.values():
        if p > 0:  # Skip zero probabilities
            entropy -= p * math.log2(p)
    return entropy

def create_tree_visualization(json_path):
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Tree Visualization')
    
    # Set global graph attributes for modern look
    dot.attr(
        rankdir='TB',  # Top to bottom layout
        splines='line',  # Straight edges instead of curved
        fontname='Arial',
        bgcolor='white'
    )
    
    # Set default node attributes
    dot.attr('node',
        shape='circle',
        style='filled',
        fillcolor='#E8E8E8',
        fontname='Arial',
        fontsize='8',    # Reduced from 11 to 8
        height='0.8',    # Increased from 0.5 to 0.8
        width='0.8',     # Increased from 0.5 to 0.8
        fixedsize='true',
        fontcolor='black'
    )
    
    # Set default edge attributes
    dot.attr('edge',
        color='#666666',
        penwidth='1.5',
        arrowsize='0.8',
        fontcolor='black'  # Added explicit black font color for edge labels
    )
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Counter for unique node IDs
    node_counter = [0]
    # List to store terminal nodes
    terminal_nodes = []
    
    def add_node_to_graph(node, parent_id=None, parent_node=None):
        current_id = str(node_counter[0])
        node_counter[0] += 1
        
        # Add parent reference to node
        node['parent'] = parent_node
        
        # Get node value (reward) and entropy
        value = node.get('value', 'N/A')
        visits = node.get('visits', 'N/A')
        
        # Calculate entropy from probability distributions in probs array
        text_entropy = 0
        if 'probs' in node:
            # For each probability distribution in probs
            entropies = []
            for prob_dist in node['probs']:
                # Filter out the special tokens (those that are 0)
                valid_probs = {k: v for k, v in prob_dist.items() if v > 0}
                if valid_probs:
                    entropies.append(calculate_entropy(valid_probs))
            
            # Calculate mean entropy if we have any valid distributions
            if entropies:
                text_entropy = sum(entropies) / len(entropies)
        
        # Add node label with entropy
        label = f"Value: {value}\nVisits: {visits}\nEntropy: {text_entropy:.2f}"
        
        # Color node based on both value and entropy
        if value != 'N/A' and text_entropy != 0:
            # Normalize entropy (assuming typical range 0-4, adjust if needed)
            normalized_entropy = min(max(float(text_entropy), 0), 2) / 2
            
            # Create a red-green gradient based on entropy
            # High entropy (1.0) -> Red (255,0,0)
            # Low entropy (0.0) -> Green (0,255,0)
            red = 100
            green = 100
            blue = int(255 * (1 - normalized_entropy))
            
            color = f"#{red:02x}{green:02x}{blue:02x}"
        else:
            color = '#E8E8E8'  # Default gray for N/A
        
        # Set node style and border based on terminal status and correctness
        node_style = 'filled'
        node_penwidth = '1.0'
        
        # For terminal nodes, check answer correctness and add thick colored border
        if node.get('terminal', False):
            node_penwidth = '4.0'  # Thick border for terminal nodes
            
            # Check answer correctness
            if node.get('answer_correct') is True:
                node_color = 'green'  # Green border for correct answers
            elif node.get('answer_correct') is False:
                node_color = 'red'    # Red border for incorrect answers
            else:
                node_color = 'black'  # Default black border if correctness is unknown
                
            # Add node to graph with custom border
            dot.node(current_id, label, 
                    fillcolor=color, 
                    style='filled,setlinewidth({})'.format(node_penwidth), 
                    color=node_color)
        else:
            # Regular node without special border
            dot.node(current_id, label, fillcolor=color)
        
        # Connect to parent if exists
        if parent_id is not None:
            dot.edge(parent_id, current_id)
        
        # If this is a terminal node, add it to our list
        if node.get('terminal', False):
            terminal_nodes.append(node)
        
        # Recursively add children
        for child in node.get('children', []):
            add_node_to_graph(child, current_id, node)
    
    # Start with root node (result)
    root = data['result']
    add_node_to_graph(root)
    
    # Create viz directory if it doesn't exist
    viz_dir = Path(json_path).parent / 'viz'
    viz_dir.mkdir(exist_ok=True)
    
    # Save visualization with the same name as the input file
    output_filename = Path(json_path).stem
    output_path = viz_dir / output_filename
    dot.render(output_path, format='png', cleanup=True)
    
    # Print conversation paths for terminal nodes
    print("\nConversation paths for terminal nodes:")
    print("=====================================")
    for i, terminal_node in enumerate(terminal_nodes, 1):
        print(f"\nPath {i}:")
        print("--------")
        paths = get_paths_to_node(root, terminal_node)
        for path in paths:
            for j, text in enumerate(path, 1):
                if text and text.strip():  # Only print if there's actual text
                    print(f"Step {j}:")
                    print(text.strip())
                    print()

def create_entropy_graph(json_path):
    """Create a graph showing entropy over steps for each reasoning path"""
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    root = data['result']
    
    # Find all terminal nodes
    terminal_nodes = []
    
    def find_terminal_nodes(node):
        if node.get('terminal', False):
            terminal_nodes.append(node)
        for child in node.get('children', []):
            find_terminal_nodes(child)
    
    find_terminal_nodes(root)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # For each terminal node, plot the entropy path
    for i, terminal_node in enumerate(terminal_nodes):
        # Get path to this terminal node
        paths = get_paths_to_node(root, terminal_node)
        
        for path in paths:
            # Collect entropy values along the path
            entropies = []
            nodes_in_path = []
            
            # Start with root and follow the path
            current = root
            nodes_in_path.append(current)
            
            # Follow the path through the tree
            for j in range(1, len(path)):
                text = path[j]
                # Find the child with matching text
                for child in current.get('children', []):
                    if child.get('text', '') == text:
                        current = child
                        nodes_in_path.append(current)
                        break
            
            # Calculate entropy for each node in the path
            for node in nodes_in_path:
                node_entropy = 0
                if 'probs' in node:
                    entropies_list = []
                    for prob_dist in node['probs']:
                        valid_probs = {k: v for k, v in prob_dist.items() if v > 0}
                        if valid_probs:
                            entropies_list.append(calculate_entropy(valid_probs))
                    
                    if entropies_list:
                        node_entropy = sum(entropies_list) / len(entropies_list)
                entropies.append(node_entropy)
            
            # Determine line color based on correctness
            if terminal_node.get('answer_correct') is True:
                line_color = 'green'
            elif terminal_node.get('answer_correct') is False:
                line_color = 'red'
            else:
                line_color = 'gray'  # Default if correctness is unknown
            
            # Plot the entropy path with normalized x-axis
            # Each path will start at 0 and end at 1.0
            total_steps = len(entropies)
            if total_steps > 1:  # Avoid division by zero
                normalized_steps = [i/(total_steps-1) for i in range(total_steps)]
                plt.plot(normalized_steps, entropies, color=line_color, alpha=0.7, linewidth=2)
                
                # Add a dot at the terminal step (which is now always at x=1.0)
                plt.scatter(1.0, entropies[-1], color=line_color, s=100, zorder=5)
    
    # Set labels and title
    plt.xlabel('Relative Reasoning Progress', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.title('Entropy During Reasoning Process', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create viz directory if it doesn't exist
    viz_dir = Path(json_path).parent / 'viz'
    viz_dir.mkdir(exist_ok=True)
    
    # Save visualization
    output_filename = Path(json_path).stem + '_viz.png'
    output_path = viz_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Entropy graph saved as {output_path}")

def create_binned_entropy_figure(json_paths):
    """
    Create a figure that bins reasoning progress, averages entropy for correct/incorrect paths,
    and plots line+scatter with shaded standard deviations.
    
    Args:
        json_paths: List of paths to JSON files to process
    """
    # Define bins for reasoning progress (0 to 1)
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Data structures to collect entropy values by bin
    correct_entropies = [[] for _ in range(num_bins)]
    incorrect_entropies = [[] for _ in range(num_bins)]
    
    # Process each JSON file
    for json_path in json_paths:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        root = data['result']
        
        # Find all terminal nodes
        terminal_nodes = []
        
        def find_terminal_nodes(node):
            if node.get('terminal', False):
                terminal_nodes.append(node)
            for child in node.get('children', []):
                find_terminal_nodes(child)
        
        find_terminal_nodes(root)
        
        # For each terminal node, process the entropy path
        for terminal_node in terminal_nodes:
            # Get path to this terminal node
            paths = get_paths_to_node(root, terminal_node)
            
            for path in paths:
                # Collect entropy values along the path
                entropies = []
                nodes_in_path = []
                
                # Start with root and follow the path
                current = root
                nodes_in_path.append(current)
                
                # Follow the path through the tree
                for j in range(1, len(path)):
                    text = path[j]
                    # Find the child with matching text
                    for child in current.get('children', []):
                        if child.get('text', '') == text:
                            current = child
                            nodes_in_path.append(current)
                            break
                
                # Calculate entropy for each node in the path
                for node in nodes_in_path:
                    node_entropy = 0
                    if 'probs' in node:
                        entropies_list = []
                        for prob_dist in node['probs']:
                            valid_probs = {k: v for k, v in prob_dist.items() if v > 0}
                            if valid_probs:
                                entropies_list.append(calculate_entropy(valid_probs))
                        
                        if entropies_list:
                            node_entropy = sum(entropies_list) / len(entropies_list)
                    entropies.append(node_entropy)
                
                # Skip if path is too short
                if len(entropies) <= 1:
                    continue
                
                # Normalize the path steps to [0, 1]
                total_steps = len(entropies)
                normalized_steps = [i/(total_steps-1) for i in range(total_steps)]
                
                # Determine if this is a correct or incorrect path
                is_correct = terminal_node.get('answer_correct', False)
                
                # Assign entropy values to bins
                for step, entropy in zip(normalized_steps, entropies):
                    # Find which bin this step belongs to
                    bin_idx = min(int(step * num_bins), num_bins - 1)
                    
                    # Add entropy to appropriate list based on correctness
                    if is_correct:
                        correct_entropies[bin_idx].append(entropy)
                    else:
                        incorrect_entropies[bin_idx].append(entropy)
    
    # Calculate statistics for each bin
    correct_means = []
    correct_stds = []
    incorrect_means = []
    incorrect_stds = []
    
    for bin_idx in range(num_bins):
        # Correct paths
        if correct_entropies[bin_idx]:
            correct_means.append(np.mean(correct_entropies[bin_idx]))
            correct_stds.append(np.std(correct_entropies[bin_idx]))
        else:
            correct_means.append(np.nan)
            correct_stds.append(np.nan)
        
        # Incorrect paths
        if incorrect_entropies[bin_idx]:
            incorrect_means.append(np.mean(incorrect_entropies[bin_idx]))
            incorrect_stds.append(np.std(incorrect_entropies[bin_idx]))
        else:
            incorrect_means.append(np.nan)
            incorrect_stds.append(np.nan)
    
    # Convert to numpy arrays for easier manipulation
    correct_means = np.array(correct_means)
    correct_stds = np.array(correct_stds)
    incorrect_means = np.array(incorrect_means)
    incorrect_stds = np.array(incorrect_stds)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot correct paths with shaded std
    plt.plot(bin_centers, correct_means, 'o-', color='green', label='Correct Paths', 
             markersize=8, linewidth=2, alpha=0.8)
    plt.fill_between(bin_centers, 
                     correct_means - correct_stds, 
                     correct_means + correct_stds, 
                     color='green', alpha=0.2)
    
    # Plot incorrect paths with shaded std
    plt.plot(bin_centers, incorrect_means, 'o-', color='red', label='Incorrect Paths', 
             markersize=8, linewidth=2, alpha=0.8)
    plt.fill_between(bin_centers, 
                     incorrect_means - incorrect_stds, 
                     incorrect_means + incorrect_stds, 
                     color='red', alpha=0.2)
    
    # Set labels and title
    plt.xlabel('Reasoning Progress', fontsize=12)
    plt.ylabel('Average Entropy', fontsize=12)
    plt.title('Average Entropy by Reasoning Progress', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Create viz directory if it doesn't exist
    output_dir = Path(json_paths[0]).parent
    viz_dir = output_dir / 'viz'
    viz_dir.mkdir(exist_ok=True)
    
    # Save visualization
    output_path = viz_dir / 'binned_entropy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Binned entropy comparison saved as {output_path}")

if __name__ == "__main__":
    # Path to the directory containing JSON files
    # output_dir = "../outputs/AIME-20250227_015351"
    
    # Use a more recent output directory or allow command line argument
    # Default to the most recent output directory if no argument provided
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Find the most recent output directory
        base_output_dir = "../outputs"
        if os.path.exists(base_output_dir):
            output_dirs = [os.path.join(base_output_dir, d) for d in os.listdir(base_output_dir) 
                          if os.path.isdir(os.path.join(base_output_dir, d))]
            if output_dirs:
                output_dir = max(output_dirs, key=os.path.getmtime)
            else:
                output_dir = base_output_dir
        else:
            output_dir = base_output_dir
            os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for JSON files in: {output_dir}")
    
    # Find all JSON files in the directory
    json_files = glob.glob(f"{output_dir}/*.json")
    
    if not json_files:
        print(f"No JSON files found in {output_dir}")
    else:
        print(f"Found {len(json_files)} JSON files to process")
        
        # Process each JSON file
        for json_path in json_files:
            print(f"\nProcessing {json_path}...")
            create_tree_visualization(json_path)
            create_entropy_graph(json_path)
            print(f"Completed processing {json_path}")
        
        # Create the binned entropy figure using all JSON files
        if len(json_files) > 0:
            print("\nCreating binned entropy comparison figure...")
            create_binned_entropy_figure(json_files)
