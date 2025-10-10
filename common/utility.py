import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

ELEMENT_COLORS = {
    3: ("#87CEEB", "Triangle"),
    4: ("#90EE90", "Quad"),
    5: ("#FFD700", "Pentagon"),
    6: ("#FFA07A", "Hexagon"),
    "other": ("#D3D3D3", "Other"),
}


def polygon_area(points):
    """Calculates the area of a polygon using the shoelace formula."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_geometry_extent(nodes):
    """Computes the extent of the geometry based on node coordinates."""
    min_coords = np.min(nodes, axis=0)
    max_coords = np.max(nodes, axis=0)
    extent = np.linalg.norm(max_coords - min_coords)
    return extent if extent > 0 else 1.0


def plot_mesh(
    ax, nodes, cells, show_nodes=False, show_cells=False, parts=None, title="Mesh"
):
    """
    Plots a 2D mesh with options for labels and partitioning colors.

    Args:
        ax: Matplotlib axes object.
        nodes (np.ndarray): Array of node coordinates (num_nodes, 2).
        cells (list): List of lists, where each inner list contains the node indices for a cell.
        show_nodes (bool): Whether to display node labels.
        show_cells (bool): Whether to display cell labels.
        parts (np.ndarray, optional): Array of partition IDs for each cell.
        title (str, optional): The title for the plot.
    """
    if nodes.shape[1] > 2:
        nodes = nodes[:, :2]

    num_nodes = nodes.shape[0]
    geometry_extent = get_geometry_extent(nodes)

    patches = []

    part_colors = None
    if parts is not None:
        unique_parts = np.unique(parts)
        num_parts = len(unique_parts)
        cmap = plt.cm.get_cmap("tab20", num_parts)
        part_min = unique_parts.min()
        part_range = (unique_parts.max() - part_min) if num_parts > 1 else 1

        def get_part_color(part_id):
            normalized_part_id = (part_id - part_min) / part_range
            return cmap(normalized_part_id)

        part_colors = {part_id: get_part_color(part_id) for part_id in unique_parts}

    for i, cell_conn in enumerate(cells):
        points = nodes[cell_conn]

        if parts is not None and part_colors is not None:
            color = part_colors[parts[i]]
        else:
            color, _ = ELEMENT_COLORS.get(len(cell_conn), ELEMENT_COLORS["other"])

        polygon = Polygon(points, facecolor=color, edgecolor="k", alpha=0.7, lw=0.5)
        patches.append(polygon)

        if show_cells:
            area = polygon_area(points)
            # Scale font size based on the element area relative to the geometry extent
            font_scale_factor = np.sqrt(area) / geometry_extent
            cell_fontsize = min(max(2, int(font_scale_factor * 120)), 10)

            cell_centroid = np.mean(points, axis=0)
            ax.text(
                cell_centroid[0],
                cell_centroid[1],
                str(i),
                color="black",
                ha="center",
                va="center",
                fontsize=cell_fontsize,
                weight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )

    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)

    if show_nodes:
        # Pre-compute node-to-cell connectivity
        node_to_cells = [[] for _ in range(num_nodes)]
        for i, cell_conn in enumerate(cells):
            for node_idx in cell_conn:
                node_to_cells[node_idx].append(i)

        for i in range(num_nodes):
            # Find neighboring nodes
            neighbor_nodes = set()
            for cell_idx in node_to_cells[i]:
                for node_idx in cells[cell_idx]:
                    if node_idx != i:
                        neighbor_nodes.add(node_idx)

            if not neighbor_nodes:
                # Default font size for isolated nodes
                node_fontsize = 8
            else:
                # Calculate average distance to neighboring nodes
                distances = np.linalg.norm(
                    nodes[list(neighbor_nodes)] - nodes[i], axis=1
                )
                avg_dist = np.mean(distances)

                # Scale font size based on the average distance relative to the geometry extent
                font_scale_factor = avg_dist / geometry_extent
                node_fontsize = min(max(2, int(font_scale_factor * 100)), 10)

            ax.text(
                nodes[i, 0],
                nodes[i, 1],
                str(i),
                color="darkred",
                ha="center",
                va="center",
                fontsize=node_fontsize,
                bbox=dict(
                    facecolor="yellow",
                    alpha=0.6,
                    edgecolor="none",
                    boxstyle="round,pad=0.1",
                ),
            )

    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("X", fontsize=14, labelpad=8)
    ax.set_ylabel("Y", fontsize=14, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", pad=2, labelsize=12)
    ax.autoscale_view()

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = []
    if parts is not None and part_colors is not None:
        unique_parts = np.unique(parts)
        for part_id in unique_parts:
            num_cells_in_part = np.sum(parts == part_id)
            legend_handles.append(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    color=part_colors[part_id],
                    label=f"Part {part_id} (#{num_cells_in_part})",
                )
            )
    else:
        # Create legend from ELEMENT_COLORS
        cell_counts = {}
        for cell in cells:
            n_sides = len(cell)
            if n_sides in ELEMENT_COLORS:
                label = ELEMENT_COLORS[n_sides][1]
            else:
                label = ELEMENT_COLORS["other"][1]
            cell_counts[label] = cell_counts.get(label, 0) + 1

        for color, label in ELEMENT_COLORS.values():
            count = cell_counts.get(label, 0)
            if count > 0:
                legend_handles.append(
                    Rectangle((0, 0), 1, 1, color=color, label=f"{label} (#{count})")
                )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=14,
        frameon=False,
        ncol=1,
    )
