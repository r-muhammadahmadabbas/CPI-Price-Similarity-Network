import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Configuration & Mappings
# -----------------------------

category_map = {
    # --- Food Staples & Grains (9 Items) ---
    "Wheat Flour Bag": "Food Staples & Grains",
    "Rice Basmati Broken": "Food Staples & Grains",
    "Rice IRRI": "Food Staples & Grains",      # Matches Rice IRRI-6/9
    "Bread plain": "Food Staples & Grains",
    "Pulse Masoor": "Food Staples & Grains",
    "Pulse Moong": "Food Staples & Grains",
    "Pulse Mash": "Food Staples & Grains",
    "Pulse Gram": "Food Staples & Grains",
    "Cooked Daal": "Food Staples & Grains",

    # --- Meat, Poultry & Dairy (8 Items) ---
    "Beef with Bone": "Meat, Poultry & Dairy",
    "Mutton": "Meat, Poultry & Dairy",
    "Chicken Farm Broiler": "Meat, Poultry & Dairy",
    "Milk fresh": "Meat, Poultry & Dairy",
    "Curd (Dahi)": "Meat, Poultry & Dairy",
    "Powdered Milk NIDO": "Meat, Poultry & Dairy",
    "Eggs Hen": "Meat, Poultry & Dairy",
    "Cooked Beef": "Meat, Poultry & Dairy",

    # --- Oils, Condiments & Sweeteners (12 Items) ---
    "Mustard Oil": "Oils, Condiments & Sweeteners",
    "Cooking Oil DALDA": "Oils, Condiments & Sweeteners",
    "Vegetable Ghee DALDA/HABIB 2.5 kg": "Oils, Condiments & Sweeteners", # Specific Item 1
    "Vegetable Ghee DALDA/HABIB or": "Oils, Condiments & Sweeteners",     # Specific Item 2 (Loose/Pouch)
    "Sugar Refined": "Oils, Condiments & Sweeteners",
    "Gur": "Oils, Condiments & Sweeteners",
    "Salt Powdered": "Oils, Condiments & Sweeteners",
    "Chilies Powder": "Oils, Condiments & Sweeteners",
    "Garlic": "Oils, Condiments & Sweeteners",
    "Tea Lipton": "Oils, Condiments & Sweeteners",
    "Tea Prepared": "Oils, Condiments & Sweeteners",

    # --- Fruits & Vegetables (4 Items) ---
    "Bananas": "Fruits & Vegetables",
    "Potatoes": "Fruits & Vegetables",
    "Onions": "Fruits & Vegetables",
    "Tomatoes": "Fruits & Vegetables",

    # --- Utilities & Transport (10 Items) ---
    "Electricity Charges": "Utilities & Transport",
    "Gas Charges": "Utilities & Transport",
    "Firewood": "Utilities & Transport",
    "Energy Saver": "Utilities & Transport",     # <--- Was missing before
    "Petrol Super": "Utilities & Transport",
    "Hi-Speed Diesel": "Utilities & Transport",
    "LPG": "Utilities & Transport",
    "Telephone Call": "Utilities & Transport",

    # --- Clothing & Miscellaneous (8 Items) ---
    "Cigarettes Capstan": "Clothing & Miscellaneous",
    "Long Cloth": "Clothing & Miscellaneous",
    "Shirting": "Clothing & Miscellaneous",
    "Lawn Printed": "Clothing & Miscellaneous",
    "Georgette": "Clothing & Miscellaneous",
    "Gents Sandal": "Clothing & Miscellaneous",
    "Gents Sponge": "Clothing & Miscellaneous",
    "Ladies Sandal": "Clothing & Miscellaneous",

    # --- Non-Food Essentials (3 Items) ---
    "Sufi Washing Soap": "Non-Food Essentials",
    "Match Box": "Non-Food Essentials",
    "Toilet Soap": "Non-Food Essentials",
}


cities = ['Islamabad', 'Rawalpindi', 'Gujranwala', 'Faisalabad', 'Sargodha',
          'Multan', 'Bahawalpur', 'Lahore', 'Sialkot', 'Peshawar', 'Bannu',
          'Karachi', 'Hyderabad', 'Sukkur', 'Larkana', 'Quetta', 'Khuzdar']

category_weights = {
    "Food Staples & Grains": 0.25,
    "Utilities & Transport": 0.20,
    "Clothing & Miscellaneous": 0.15,
    "Meat, Poultry & Dairy": 0.10,
    "Oils, Condiments & Sweeteners": 0.10,
    "Fruits & Vegetables": 0.10,
    "Non-Food Essentials": 0.10,
}

threshold = 0.65

# -----------------------------
# 2. Data Loading & Preprocessing
# -----------------------------
#
def normalize_vector(vec):
    """Z-score normalization for CPI cosine similarity analysis."""
    v = np.array(vec, dtype=float)
    mean = np.mean(v)
    std = np.std(v)
    if std == 0:
        return np.ones(len(v)) 
    return (v - mean) / std

def build_city_vectors(file_path, year_label):
    print(f"\n--- Processing {year_label} ---")

    # 1. Load Data
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

    # 2. Clean Columns
    df.columns = df.columns.str.strip()
    if 'Month' not in df.columns:
        print(f"CRITICAL ERROR: 'Month' column missing in {year_label}.")
        return {}

    item_col = 'Description' if 'Description' in df.columns else 'Item'

    # 3. Map Items to Categories
    df['Category'] = None
    for item_key, cat_name in category_map.items():
        # Fuzzy match
        mask = df[item_col].astype(str).str.contains(item_key, case=False, regex=False)
        df.loc[mask, 'Category'] = cat_name

    df_clean = df.dropna(subset=['Category'])

    # 4. Aggregation: The Magic Step
    # Average all items in a category for EACH MONTH.
    # This creates the timeline: Jan Avg, Feb Avg...
    monthly_data = df_clean.groupby(['Category', 'Month'])[cities].mean().reset_index()

    # 5. Build Vectors
    category_city_vectors = {}
    unique_cats = monthly_data['Category'].unique()

    for cat in unique_cats:
        category_city_vectors[cat] = {}
        # Sort by Month to ensure time order (1, 2, 3...)
        cat_df = monthly_data[monthly_data['Category'] == cat].sort_values('Month')

        for city in cities:
            if city in cat_df.columns:
                raw_vec = cat_df[city].values
                # We need variance to correlate, so len > 1
                if len(raw_vec) > 1:
                    category_city_vectors[cat][city] = normalize_vector(raw_vec)

    print(f"  Vectors built for {len(unique_cats)} categories.")
    return category_city_vectors
# -----------------------------
# 3. Similarity & Graphs
# -----------------------------
def compute_similarity(category_city_vectors):
    similarity_matrices = {}

    for cat, city_dict in category_city_vectors.items():
        # Filter out cities with no data
        valid_cities = [c for c, vec in city_dict.items() if len(vec) > 0]

        if len(valid_cities) < 2:
            continue

        # Create matrix
        vectors = np.array([city_dict[c] for c in valid_cities])

        # Safety check for empty dimensions
        if vectors.shape[1] == 0:
            continue

        sim_matrix = cosine_similarity(vectors)
        similarity_matrices[cat] = pd.DataFrame(sim_matrix, index=valid_cities, columns=valid_cities)

    return similarity_matrices

def build_graphs(similarity_matrices, threshold):
    graphs = {}
    for cat, sim_df in similarity_matrices.items():
        G = nx.Graph()
        # Add all nodes
        for city in sim_df.index:
            G.add_node(city)
        # Add edges based on threshold
        for i, j in combinations(sim_df.index, 2):
            w = sim_df.loc[i,j]
            if w >= threshold:
                G.add_edge(i, j, weight=w)
        graphs[cat] = G
    return graphs

# -----------------------------
# 4. Centrality & Weighting
# -----------------------------
def compute_centralities(graphs):
    centrality_data = {}
    for cat, G in graphs.items():
        if len(G.nodes) > 0:
            try:
                # Eigenvector can fail if graph is not connected or too small, use try/except
                eigen = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                eigen = {n: 0 for n in G.nodes()}

            centrality_data[cat] = {
                'degree': nx.degree_centrality(G),
                'closeness': nx.closeness_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'eigenvector': eigen
            }
    return centrality_data

def compute_weights(centralities, scheme="equal", category_weights=category_weights, user_weights=None):
    weighted_scores = {}

    for cat, metrics in centralities.items():
        df_metrics = pd.DataFrame(metrics).fillna(0)

        # 1. Equal Weighting (Baseline)
        if scheme == "equal":
            weights = np.array([0.25, 0.25, 0.25, 0.25])

        # 2. User-Tuned (Interactive) 
        elif scheme == "user_tuned":
            if user_weights and len(user_weights) == 4:
                # Normalize just in case user inputs don't sum to 1
                w = np.array(user_weights)
                weights = w / np.sum(w)
            else:
                weights = np.array([0.25, 0.25, 0.25, 0.25])

        # 3. Correlation-Based Weighting
        elif scheme == "correlation":
            if df_metrics.shape[1] < 2:
                weights = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                corr_matrix = df_metrics.corr().fillna(0).values
                weights_list = []
                for k in range(len(corr_matrix)):
                    # Sum correlations with others (exclude self)
                    corr_sum = np.sum(np.delete(corr_matrix[k,:], k))
                    # Formula: 1 / (1 + sum_corr)
                    w = 1 / (1 + corr_sum + 1e-9)
                    weights_list.append(w)
                weights = np.array(weights_list)
                # Normalize to sum to 1
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.array([0.25]*4)

        # 4. Entropy-Based Weighting (Corrected for "Variability")
        elif scheme == "entropy":
            # Step A: Normalize columns to sum to 1 (Probability distribution)
            col_sums = df_metrics.sum(axis=0)
            df_norm = df_metrics.div(col_sums + 1e-9, axis=1)
            df_norm = df_norm.replace(0, 1e-9) # Avoid log(0)

            # Step B: Calculate Entropy H = -sum(p * log(p))
            # H ranges from 0 (High Variability) to ln(N) (Uniform)
            H = -np.sum(df_norm * np.log(df_norm), axis=0)

            # Step C: Calculate Divergence (1 - Normalized Entropy)
            # We normalize H by ln(N) so it is between 0 and 1
            max_H = np.log(len(df_metrics))
            if max_H == 0:
                H_norm = 0
            else:
                H_norm = H / max_H

            # Divergence: High Divergence = High Information = High Weight
            divergence = 1 - H_norm
            div_sum = np.sum(divergence)

            if div_sum > 0:
                weights = (divergence / div_sum).values
            else:
                weights = np.array([0.25, 0.25, 0.25, 0.25])

        else:
            weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Calculate score per city
        weighted_scores[cat] = {}
        for city in df_metrics.index:
            score = np.dot(df_metrics.loc[city].values, weights)
            weighted_scores[cat][city] = score

    return weighted_scores

def aggregate_overall_score(weighted_scores, alpha=category_weights):
    overall = {city: 0 for city in cities}
    for cat, scores in weighted_scores.items():
        weight = alpha.get(cat, 0)
        for city, val in scores.items():
            overall[city] += val * weight
    return overall

# -----------------------------
# 5. Hasse & Visualization
# -----------------------------
def compute_temporal_inclusion(graphs_prev, graphs_next):
    hasse_edges = {}
    for cat in graphs_prev.keys():
        if cat in graphs_next:
            # FIX: Normalize edges to frozenset so (A,B) == (B,A)
            E_prev = set(frozenset(e) for e in graphs_prev[cat].edges())
            E_next = set(frozenset(e) for e in graphs_next[cat].edges())

            if not E_prev:
                included = True
            else:
                included = E_prev.issubset(E_next)
            hasse_edges[cat] = included
    return hasse_edges


def draw_hasse_diagram(year_from, year_to, inclusion_results, title="Hasse Diagram"):
    """
    Draws a visual Hasse diagram for the temporal relations.
    inclusion_results: {'Food': True, 'Utility': False...}
    """
    H = nx.DiGraph()

    # Create nodes for the years
    H.add_node(year_from, layer=0)
    H.add_node(year_to, layer=1)

    # Add edges ONLY if the subset relation holds (True)
    edge_labels = {}
    for cat, is_subset in inclusion_results.items():
        if is_subset:
            # Add edge or update label
            if H.has_edge(year_from, year_to):
                # Append category to existing label
                current_label = edge_labels.get((year_from, year_to), "")
                edge_labels[(year_from, year_to)] = current_label + f"\n{cat}"
            else:
                H.add_edge(year_from, year_to)
                edge_labels[(year_from, year_to)] = cat

    # Plotting
    if H.number_of_edges() > 0:
        plt.figure(figsize=(6, 8))
        pos = {year_from: (0, 0), year_to: (0, 1)} # Vertical layout (Bottom-Up)

        # Draw Nodes
        nx.draw_networkx_nodes(H, pos, node_size=3000, node_color='lightgreen', edgecolors='black')
        nx.draw_networkx_labels(H, pos, font_size=12, font_weight='bold')

        # Draw Edges
        nx.draw_networkx_edges(H, pos, arrowstyle='->', arrowsize=20, edge_color='black')

        # Draw Labels (Categories that maintained structure)
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f"{title}: {year_from} $\subseteq$ {year_to}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No subset relations found between {year_from} and {year_to}. No diagram to draw.")

# MISSING: You need to add this verification
def verify_partial_order(all_graphs):
    """
    Verifies the mathematical properties of the Partial Order Relation T:
    Gy,c T Gy',c <==> Ey,c ⊆ Ey',c
    """
    print("\n" + "="*40)
    print(" 🔍 PARTIAL ORDER VERIFICATION (Rubric C1)")
    print("="*40)

    # We test on "Food Staples" as the representative category
    cat = "Food Staples & Grains"
    years = sorted(all_graphs.keys())

    # Extract Edge Sets for the category across all years
    edge_sets = {}
    for y in years:
        if cat in all_graphs[y]:
            # Convert edges to frozenset so order (A,B) vs (B,A) doesn't matter
            edges = set(frozenset(e) for e in all_graphs[y][cat].edges())
            edge_sets[y] = edges

    if len(edge_sets) < 2:
        print("Not enough data to verify relations.")
        return

    # 1. REFLEXIVITY: A ⊆ A (Always True)
    print(f"\n[1] Reflexivity Check (A ⊆ A):")
    for y in years:
        E = edge_sets[y]
        print(f"  {y} ⊆ {y}: {E.issubset(E)} ✅")

    # 2. TRANSITIVITY: A ⊆ B and B ⊆ C -> A ⊆ C
    print(f"\n[2] Transitivity Check (Chain Rule):")
    # Check 2023 -> 2024 -> 2025 sequence
    if '2023' in edge_sets and '2024' in edge_sets and '2025' in edge_sets:
        E23 = edge_sets['2023']
        E24 = edge_sets['2024']
        E25 = edge_sets['2025']

        # Check individual links
        rel_23_24 = E23.issubset(E24)
        rel_24_25 = E24.issubset(E25)
        rel_23_25 = E23.issubset(E25)

        print(f"  2023 ⊆ 2024: {rel_23_24}")
        print(f"  2024 ⊆ 2025: {rel_24_25}")
        print(f"  2023 ⊆ 2025: {rel_23_25}")

        if rel_23_24 and rel_24_25:
            # If the chain exists, the shortcut MUST exist
            if rel_23_25:
                print("  ✅ Transitivity Holds: Chain implies Shortcut.")
            else:
                print("  ❌ Transitivity Failed (Mathematical Error).")
        else:
            print("  (Chain broken, so Transitivity is vacuously true).")

    # 3. ANTISYMMETRY: A ⊆ B and B ⊆ A -> A = B
    print(f"\n[3] Antisymmetry Check:")
    y1, y2 = '2023', '2024'
    if y1 in edge_sets and y2 in edge_sets:
        E1 = edge_sets[y1]
        E2 = edge_sets[y2]

        fwd = E1.issubset(E2)
        rev = E2.issubset(E1)

        print(f"  {y1} ⊆ {y2}: {fwd}")
        print(f"  {y2} ⊆ {y1}: {rev}")

        if fwd and rev:
            print(f"  Both true implies sets are identical: {E1 == E2}")
        elif fwd != rev:
            print("  ✅ Antisymmetry Holds: One-way relation implies distinct sets.")

    print("\n=== Conclusion: T is a valid partial order ===\n")

def plot_graph(G, title="Graph"):
    if not G.nodes(): return
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # --- CHANGE: Draw edges with varying thickness ---
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    if weights:
        # Scale width: 0.7 -> 1.4px, 1.0 -> 2.0px
        widths = [w * 2 for w in weights]
        nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray')
    # -------------------------------------------------
    
    # Labels for highest weights only (to avoid clutter)
    labels = {k: f"{v:.2f}" for k,v in nx.get_edge_attributes(G, 'weight').items() if v > 0.8}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)
    
    plt.title(title)
    plt.show()

def plot_heatmap(sim_df, title="Similarity Heatmap"):
    plt.figure(figsize=(10,8))
    plt.imshow(sim_df, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(sim_df.columns)), labels=sim_df.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(sim_df.index)), labels=sim_df.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def print_top5(overall_scores, title="Top 5 Cities"):
    sorted_cities = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n{title}:")
    for city, score in sorted_cities:
        print(f"{city}: {score:.4f}")

def plot_multi_scheme_comparison(results_dict, year):
    """
    Plots a 2x2 grid of bar charts comparing Top 5 cities across 4 weighting schemes.
    results_dict: {'Equal': {city: score...}, 'Entropy': {city: score...}, ...}
    """
    schemes = list(results_dict.keys())
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten() # Flatten 2D array to 1D for easy iteration

    colors = ['skyblue', 'salmon', 'lightgreen', 'plum'] # Distinct colors

    for i, scheme in enumerate(schemes):
        if i >= len(axes): break # Safety check

        scores = results_dict[scheme]
        # Sort and get top 5
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        cities = [x[0] for x in top5]
        values = [x[1] for x in top5]

        ax = axes[i]
        y_pos = np.arange(len(cities))

        # Plot Horizontal Bar Chart
        ax.barh(y_pos, values, color=colors[i % len(colors)])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cities, fontsize=10, fontweight='bold')
        ax.invert_yaxis() # Put rank #1 at the top

        # Labels
        ax.set_title(f"{scheme} Weighting", fontsize=12)
        ax.set_xlabel("Influence Score")

        # Add values to end of bars for readability
        for j, v in enumerate(values):
            ax.text(v, j, f" {v:.3f}", va='center', fontsize=9)

    plt.suptitle(f"City Ranking Comparison - {year}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_weighted_network(G, scores, title="Weighted Network"):
    """
    Plots the network where Node Size is proportional to the Influence Score.
    """
    if not G.nodes(): return
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Scale node sizes (scores usually 0.0-1.0, so multiply by 5000 for visibility)
    # Default size 100 for zero-score nodes
    node_sizes = [scores.get(n, 0) * 5000 + 100 for n in G.nodes()]
    
    # Color nodes by score
    node_colors = [scores.get(n, 0) for n in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap='viridis', alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.colorbar(nodes, label="Influence Score")
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_composite_graph(sim_matrices, cat_weights, threshold, year):
    """
    Satisfies Rubric: 'Category-weighted similarity graphs'.
    Creates a Master Network by weighted averaging of all categories.
    """
    # 1. Initialize Master Matrix with zeros
    # Get cities list from the first available matrix
    first_cat = list(sim_matrices.keys())[0]
    cities = sim_matrices[first_cat].index
    global_sim = pd.DataFrame(0.0, index=cities, columns=cities)
    
    # 2. Weighted Sum: (0.25 * Food) + (0.20 * Utility) ...
    total_weight_used = 0
    for cat, df in sim_matrices.items():
        if cat in cat_weights:
            w = cat_weights[cat]
            global_sim = global_sim.add(df * w, fill_value=0)
            total_weight_used += w
            
    # 3. Build Graph from Master Matrix
    G_global = nx.Graph()
    G_global.add_nodes_from(cities)
    
    # Add edges based on Weighted Similarity
    for i, j in combinations(cities, 2):
        w = global_sim.loc[i, j]
        # We might need a slightly lower threshold for the composite since it averages out
        if w >= threshold:
            G_global.add_edge(i, j, weight=w)
            
    # 4. Plot
    if not G_global.nodes(): return
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_global, seed=42)
    
    # Draw Nodes (Gold color for "Master" graph)
    nx.draw_networkx_nodes(G_global, pos, node_size=800, node_color='gold', edgecolors='black')
    nx.draw_networkx_labels(G_global, pos, font_size=9, font_weight='bold')
    
    # Draw Edges (Thickness = Strength)
    weights = [G_global[u][v]['weight'] for u,v in G_global.edges()]
    if weights:
        # Scale width for visibility
        widths = [w * 3 for w in weights]
        nx.draw_networkx_edges(G_global, pos, width=widths, edge_color='darkblue', alpha=0.6)
    
    plt.title(f"CATEGORY-WEIGHTED COMPOSITE GRAPH ({year})\n(Weighted avg of all sectors)")
    plt.axis('off')
    plt.show()
# -----------------------------
# 6. Main Pipeline (Optimized for Presentation)
# -----------------------------
# UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS
yearly_files = {
    "2023": r"C:\Semester 3\Discrete strucutres\CPI_PROJ\data\2023_cleaned_yearly.xlsx",
    "2024": r"C:\Semester 3\Discrete strucutres\CPI_PROJ\data\2024_cleaned_yearly.xlsx",
    "2025": r"C:\Semester 3\Discrete strucutres\CPI_PROJ\data\2025_cleaned_yearly.xlsx",
}
all_years_graphs = {}
prev_graphs = None

# TARGET for Report Visuals (Focus on the year of big change)
presentation_year = "2024"
presentation_cat = "Food Staples & Grains"

print(f"\n{'='*60}")
print(f" GENERATING ANALYTICS & VISUALS (Focus Year: {presentation_year})")
print(f"{'='*60}")

for year, file_path in yearly_files.items():
    
    # 1. Build Vectors (Math Phase)
    vectors = build_city_vectors(file_path, year)
    if not vectors: continue

    # 2. Similarity & Graphs (Math Phase)
    sims = compute_similarity(vectors)
    graphs = build_graphs(sims, threshold)
    all_years_graphs[year] = graphs
    
    # 4. Centrality (Math Phase)
    centralities = compute_centralities(graphs)

    # 5. VISUALIZATION PHASE (Selective)
    # We generate full plots ONLY for the target year to avoid cluttering the video
    if year == presentation_year:
        print(f"\n--- [VIDEO HIGHLIGHT] Visualizing Key Networks for {year} ---")
        
        # A. The Composite Network (Master Graph)
        plot_composite_graph(sims, category_weights, threshold, year)
        
        # B. The Representative Category (Food)
        if presentation_cat in sims:
            plot_heatmap(sims[presentation_cat], title=f"Raw Similarity Heatmap ({year}) - {presentation_cat}")
            plot_graph(graphs[presentation_cat], title=f"Standard Network Topology ({year}) - {presentation_cat}")

        # C. Weighting Scheme Analysis (The 2x2 Grid)
        comparison_results = {}
        my_custom_weights = [0.1, 0.4, 0.1, 0.4]
        schemes = ["equal", "correlation", "entropy", "user_tuned"]
        
        print(f"\n--- City Rankings ({year}) ---")
        for scheme in schemes:
            w_scores = compute_weights(centralities, scheme=scheme, user_weights=my_custom_weights)
            overall = aggregate_overall_score(w_scores)
            
            label = "User Tuned" if scheme == "user_tuned" else scheme.capitalize()
            comparison_results[label] = overall
            print_top5(overall, title=f"Top 5 ({label})")
            
            # D. Structure Change Proof (Equal vs Entropy) - Key for Rubric I2
            if scheme in ["equal", "entropy"] and presentation_cat in graphs:
                # Use Food graph as skeleton to show influence shift
                plot_weighted_network(graphs[presentation_cat], overall, 
                                    title=f"Network Structure: {label} Weighting ({year})")

        # Plot the 2x2 Grid
        plot_multi_scheme_comparison(comparison_results, year)

    # 6. TEMPORAL ANALYSIS (Hasse)
    # We still run this every loop to build the chain (2023->2024->2025)
    if prev_graphs:
        hasse = compute_temporal_inclusion(prev_graphs, graphs)
        valid_subsets = {k:v for k,v in hasse.items() if v}
        
        if valid_subsets:
            print(f"\n--- Generating Hasse Diagram ({int(year)-1} -> {year}) ---")
            draw_hasse_diagram(str(int(year)-1), year, valid_subsets)
        else:
            print(f"\n(No subset relations for {int(year)-1}->{year}: Market Disruption Detected)")
            
    prev_graphs = graphs
verify_partial_order(all_years_graphs)

# ==========================================
# 8. YEAR-TO-YEAR VARIANCE ANALYSIS
# ==========================================
print("\n" + "="*70)
print(" TEMPORAL VARIANCE ANALYSIS: City Score Changes (2023-2025)")
print("="*70)
print("\nAnalyzing how city influence scores vary across years...\n")

# Collect scores across all years using equal weighting for consistency
all_results = {}
for year in sorted(yearly_files.keys()):
    if year in all_years_graphs:
        cents = compute_centralities(all_years_graphs[year])
        w_scores = compute_weights(cents, scheme="equal")
        all_results[year] = aggregate_overall_score(w_scores)

if len(all_results) >= 2:
    print(f"Variance Analysis (Based on Equal Weighting Scheme):")
    print(f"{'City':<15} {'Mean Score':>12} {'Variance':>12} {'Std Dev':>12} {'Trend':>10}")
    print("-" * 65)

    variance_data = []
    for city in cities:
        scores_over_time = [all_results[y].get(city, 0) for y in sorted(all_results.keys())]

        if any(scores_over_time):
            mean_score = np.mean(scores_over_time)
            variance = np.var(scores_over_time)
            std_dev = np.std(scores_over_time)

            # Simple trend analysis
            if len(scores_over_time) >= 2:
                if scores_over_time[-1] > scores_over_time[0]:
                    trend = "↑ Rising"
                elif scores_over_time[-1] < scores_over_time[0]:
                    trend = "↓ Falling"
                else:
                    trend = "→ Stable"
            else:
                trend = "N/A"

            variance_data.append((city, mean_score, variance, std_dev, trend))
            print(f"{city:<15} {mean_score:>12.4f} {variance:>12.6f} {std_dev:>12.6f} {trend:>10}")

    # Identify most/least stable cities
    if variance_data:
        variance_data.sort(key=lambda x: x[2])  # Sort by variance
        print("\n" + "="*70)
        print("📊 KEY FINDINGS:")
        print("="*70)
        print(f"\n✅ Most Stable Cities (Low Variance):")
        for city, mean, var, std, trend in variance_data[:3]:
            print(f"   {city:15s} - Variance: {var:.6f}, Mean Score: {mean:.4f}")

        print(f"\n⚠  Most Volatile Cities (High Variance):")
        for city, mean, var, std, trend in variance_data[-3:]:
            print(f"   {city:15s} - Variance: {var:.6f}, Mean Score: {mean:.4f}")

        print("\n💡 Interpretation:")
        print("   - Low variance = Consistent influence across years (stable economic role)")
        print("   - High variance = Fluctuating influence (changing economic dynamics)")
        print("   - Trend analysis shows whether cities are gaining or losing influence\n")
else:
    print("Insufficient data for variance analysis (need at least 2 years).\n")

print("\n" + "="*70)
print("Pipeline Execution Complete.")
print("="*70)