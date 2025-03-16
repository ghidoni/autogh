import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt


# Define the grouping function with sliders
def create_grouping_function_cl(df, num_groups):
    """
    Creates an interactive widget to group data by score ranges.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'score' and 'label' columns
    num_groups : int
        Number of groups to create

    Returns:
    --------
    None
        Displays interactive widgets
    """
    # Validate inputs
    if "score" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'score' and 'label' columns")
    if num_groups < 2:
        raise ValueError("Number of groups must be at least 2")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    # Initialize the sliders and output display
    sliders = []
    table_output = widgets.Output()
    chart_output = widgets.Output()
    bins_output = widgets.Output()

    # Create sliders with evenly distributed initial values
    score_min = df["score"].min()
    score_max = df["score"].max()
    score_range = score_max - score_min
    quantiles = np.linspace(score_min, score_max, num_groups)
    last_quantile = np.quantile(df["score"], 0.999)
    for i in range(num_groups - 1):
        initial_value = (
            quantiles[i + 1] if i < num_groups - 2 else last_quantile
        )
        slider = widgets.FloatSlider(
            value=initial_value,
            min=score_min,
            max=score_max,
            step=0.01,
            description=f"GH {i+1} Max:",
            continuous_update=False,
        )
        sliders.append(slider)

    # Function to update the output table
    def update_table(change):
        # Get all slider values and sort them
        boundary_values = (
            [score_min] + [slider.value for slider in sliders] + [score_max]
        )
        boundary_values.sort()  # Ensure boundaries are in order

        # Versao GPT
        # Assign groups using pd.cut()
        df["GH"] = pd.cut(
            df["score"],
            bins=boundary_values,
            labels=[f"GH {i+1}" for i in range(num_groups)],
            include_lowest=True,
        )

        # Create summary table
        summary = (
            df.groupby("GH")["label"]
            .agg(QTD="count", Perf="mean")
            .reset_index()
        )

        with table_output:
            table_output.clear_output()
            # Display the summary table
            display(HTML(summary.to_html(index=False)))

        with chart_output:
            chart_output.clear_output()
            plt.close()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(
                df["score"],
                bins=bins_slider.value,
                ax=ax,
                color="aqua",
                edgecolor="black",
                stat="proportion",
                kde=True,
            )
            # Add vertical lines (excluding first and last boundaries)
            for boundary in boundary_values[1:-1]:
                ax.axvline(
                    x=boundary,
                    color="red",
                    linestyle="--",
                    linewidth=0.9,
                )
            ax.set_title("Distribuição de Score com Limites de Grupo")
            ax.set_xlabel("Score")
            ax.set_ylabel("%QTD")
            plt.tight_layout()
            plt.show()

    # Attach the update function to each slider
    for slider in sliders:
        slider.observe(update_table, names="value")

    # Add controls for visualization
    bins_slider = widgets.IntSlider(
        value=50, min=5, max=100, step=5, description="Bins Hist."
    )
    bins_slider.observe(update_table, names="value")

    # Add Reset Button and Instructions
    reset_button = widgets.Button(description="Resetar Limites")

    def reset_boundaries(b):
        for i, slider in enumerate(sliders):
            slider.value = score_min + (i + 1) * (score_range / num_groups)

    reset_button.on_click(reset_boundaries)

    bins_button = widgets.Button(description="Exportar cortes")

    def export_bins(b):
        boundaries = [slider.value for slider in sliders]
        boundaries.insert(0, score_min)
        boundaries.append(score_max)
        with bins_output:
            bins_output.clear_output()
            display(HTML(f"<p><b>Cortes:</b> {boundaries}</p>"))

    bins_button.on_click(export_bins)

    # Add instructions
    instructions = widgets.HTML(
        value="<p><b>Ajuste os limites para definir os grupos.</b></p>"
    )

    table_title = widgets.HTML(value="<p><b>Tabela de volume e perf.</b></p>")

    update_table(None)

    controls_box = widgets.VBox(
        [
            instructions,
            widgets.VBox(sliders),
            reset_button,
            bins_slider,
            bins_button,
            bins_output,
        ]
    )
    table_box = widgets.VBox([table_title, table_output])
    display(widgets.HBox([controls_box, table_box, chart_output]))
