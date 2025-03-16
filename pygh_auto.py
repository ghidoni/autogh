import numpy as np
import random
from copy import deepcopy


def calculate_group_statistics(data, group_assignments):
    """
    Calculate intra-group similarity and inter-group similarity
    Higher intra and lower inter is better
    """
    groups = {}
    for idx, group_id in enumerate(group_assignments):
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(data[idx])

    # Calculate average intra-group similarity
    intra_similarity = 0
    for group_id, members in groups.items():
        group_arr = np.array(members)
        if len(group_arr) <= 1:
            continue

        # Using correlation as similarity measure
        corr_matrix = np.corrcoef(group_arr)
        # Remove self-correlations
        np.fill_diagonal(corr_matrix, 0)
        # Average correlation within group
        intra_similarity += np.sum(corr_matrix) / (
            corr_matrix.size - len(corr_matrix)
        )

    intra_similarity /= len(groups)

    # Calculate average inter-group similarity
    inter_similarity = 0
    comparisons = 0
    group_ids = list(groups.keys())

    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            group1 = np.array(groups[group_ids[i]])
            group2 = np.array(groups[group_ids[j]])

            # Calculate all pairwise correlations between groups
            inter_corr = np.mean(
                [
                    np.corrcoef(item1, item2)[0, 1]
                    for item1 in group1
                    for item2 in group2
                ]
            )
            inter_similarity += inter_corr
            comparisons += 1

    inter_similarity /= max(comparisons, 1)

    # Objective: maximize intra-group similarity and minimize inter-group similarity
    quality_score = intra_similarity - inter_similarity

    return quality_score, intra_similarity, inter_similarity


def iterative_group_optimization(data, n_groups, iterations=1000):
    """
    Optimize groups by iteratively changing boundaries
    """
    # Sort data by score
    sorted_indices = np.argsort([x[0] for x in data])
    sorted_data = [data[i] for i in sorted_indices]

    # Initial grouping - equal size groups
    n_samples = len(data)
    group_size = n_samples // n_groups

    # Initial cutting points
    cut_points = [i * group_size for i in range(1, n_groups)]

    # Create initial group assignments
    group_assignments = []
    for i in range(n_samples):
        group_id = 0
        for j, cut in enumerate(cut_points):
            if i >= cut:
                group_id = j + 1
        group_assignments.append(group_id)

    # Calculate initial score
    best_score, intra_sim, inter_sim = calculate_group_statistics(
        sorted_data, group_assignments
    )
    best_cut_points = deepcopy(cut_points)

    print(
        f"Initial score: {best_score:.4f} (intra: {intra_sim:.4f}, inter: {inter_sim:.4f})"
    )

    # Iterative optimization
    for iter in range(iterations):
        # Choose a random cutting point to modify
        cut_idx = random.randint(0, len(cut_points) - 1)

        # Modify the cutting point
        direction = random.choice([-1, 1])
        new_cut_points = deepcopy(cut_points)

        # Make sure we don't create invalid bounds
        if cut_idx == 0:
            min_val = 1
        else:
            min_val = new_cut_points[cut_idx - 1] + 1

        if cut_idx == len(cut_points) - 1:
            max_val = n_samples - 1
        else:
            max_val = new_cut_points[cut_idx + 1] - 1

        # Apply change if valid
        new_cut_point = new_cut_points[cut_idx] + direction
        if min_val <= new_cut_point <= max_val:
            new_cut_points[cut_idx] = new_cut_point

            # Create new group assignments
            new_group_assignments = []
            for i in range(n_samples):
                group_id = 0
                for j, cut in enumerate(new_cut_points):
                    if i >= cut:
                        group_id = j + 1
                new_group_assignments.append(group_id)

            # Calculate new score
            new_score, new_intra, new_inter = calculate_group_statistics(
                sorted_data, new_group_assignments
            )

            # Update if better
            if new_score > best_score:
                best_score = new_score
                best_cut_points = deepcopy(new_cut_points)
                intra_sim, inter_sim = new_intra, new_inter
                cut_points = new_cut_points

                if iter % 50 == 0:
                    print(
                        f"Iteration {iter}: Score improved to {best_score:.4f} (intra: {intra_sim:.4f}, inter: {inter_sim:.4f})"
                    )

    # Final group assignments
    final_group_assignments = []
    for i in range(n_samples):
        group_id = 0
        for j, cut in enumerate(best_cut_points):
            if i >= cut:
                group_id = j + 1
        final_group_assignments.append(group_id)

    # Restore original order
    original_order_assignments = [0] * n_samples
    for i, idx in enumerate(sorted_indices):
        original_order_assignments[idx] = final_group_assignments[i]

    print(
        f"Final score: {best_score:.4f} (intra: {intra_sim:.4f}, inter: {inter_sim:.4f})"
    )
    print(f"Final cut points: {best_cut_points}")

    return original_order_assignments, best_cut_points, best_score


# dynamic


def optimize_number_of_groups(
    data, min_groups=2, max_groups=10, iterations_per_group=500
):
    """
    Find optimal number of groups by comparing results from different numbers
    """
    best_overall_score = float("-inf")
    best_overall_assignments = None
    best_overall_n_groups = None
    best_overall_cuts = None

    for n_groups in range(min_groups, max_groups + 1):
        print(f"\nTrying {n_groups} groups:")
        assignments, cut_points, score = iterative_group_optimization(
            data, n_groups, iterations=iterations_per_group
        )

        # Consider a penalty for more groups (optional)
        # Prevents overfitting with too many groups
        complexity_penalty = 0.05 * n_groups
        adjusted_score = score - complexity_penalty

        print(
            f"Adjusted score for {n_groups} groups: {adjusted_score:.4f} (raw: {score:.4f}, penalty: {complexity_penalty:.4f})"
        )

        if adjusted_score > best_overall_score:
            best_overall_score = adjusted_score
            best_overall_assignments = assignments
            best_overall_n_groups = n_groups
            best_overall_cuts = cut_points

    print(f"\nOptimal number of groups: {best_overall_n_groups}")
    print(f"Best score: {best_overall_score:.4f}")

    return best_overall_assignments, best_overall_n_groups, best_overall_cuts


# silhouete metric
def silhouette_score_custom(data, group_assignments):
    """
    Calculate a custom silhouette score for the grouping
    Higher values indicate better clustering
    """
    from sklearn.metrics import pairwise_distances
    import numpy as np

    # Convert data and assignments to numpy arrays
    X = np.array(data)
    labels = np.array(group_assignments)

    # Calculate pairwise distances
    distances = pairwise_distances(X)

    # Calculate silhouette score
    n_samples = len(X)
    unique_labels = np.unique(labels)

    # For each sample
    silhouette_vals = []
    for i in range(n_samples):
        # Get distances to all points in the same cluster
        same_cluster_indices = np.where(labels == labels[i])[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]

        if len(same_cluster_indices) == 0:
            # If singleton cluster, silhouette is 0
            silhouette_vals.append(0)
            continue

        # Average distance to points in same cluster (a)
        a = np.mean([distances[i, idx] for idx in same_cluster_indices])

        # Average distance to points in other clusters (b)
        b_values = []
        for label in unique_labels:
            if label != labels[i]:
                other_cluster_indices = np.where(labels == label)[0]
                if len(other_cluster_indices) > 0:
                    avg_dist = np.mean(
                        [distances[i, idx] for idx in other_cluster_indices]
                    )
                    b_values.append(avg_dist)

        if not b_values:
            silhouette_vals.append(0)
            continue

        # Minimum average distance to other clusters
        b = min(b_values)

        # Calculate silhouette
        s = (b - a) / max(a, b)
        silhouette_vals.append(s)

    return np.mean(silhouette_vals)


# other metrics
from sklearn.metrics import silhouette_score

# Higher values (closer to 1) indicate better-defined clusters
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score}")

from sklearn.metrics import davies_bouldin_score

# Lower values indicate better clustering
db_index = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_index}")

from sklearn.metrics import calinski_harabasz_score

# Higher values indicate better clustering
ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {ch_score}")


# Calculate intra-cluster similarity
def intra_cluster_similarity(X, labels):
    n_clusters = len(set(labels))
    intra_sim = 0

    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            # Calculate average pairwise similarity within cluster
            from sklearn.metrics.pairwise import cosine_similarity

            sim_matrix = cosine_similarity(cluster_points)
            # Remove self-similarity (diagonal)
            np.fill_diagonal(sim_matrix, 0)
            # Average similarity
            intra_sim += sim_matrix.sum() / (sim_matrix.size - len(sim_matrix))

    return intra_sim / n_clusters


# Calculate inter-cluster similarity
def inter_cluster_similarity(X, labels):
    n_clusters = len(set(labels))
    inter_sim = 0
    comparisons = 0

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == i]
            cluster_j = X[labels == j]

            # Calculate average similarity between clusters
            from sklearn.metrics.pairwise import cosine_similarity

            sim_matrix = cosine_similarity(cluster_i, cluster_j)
            inter_sim += sim_matrix.mean()
            comparisons += 1

    return inter_sim / comparisons if comparisons > 0 else 0


# Calculate ratio of intra to inter similarity (higher is better)
intra_sim = intra_cluster_similarity(X, labels)
inter_sim = inter_cluster_similarity(X, labels)
ratio = intra_sim / inter_sim if inter_sim > 0 else float("inf")
print(f"Intra-cluster similarity: {intra_sim}")
print(f"Inter-cluster similarity: {inter_sim}")
print(f"Ratio (intra/inter): {ratio}")
