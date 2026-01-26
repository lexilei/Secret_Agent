"""
Chart generation for benchmark reports using Plotly.

Creates interactive visualizations for experiment comparisons.
"""

from typing import Dict, List, Any, Optional
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..metrics.aggregator import AggregatedMetrics


class ChartGenerator:
    """
    Generate interactive Plotly charts for benchmark reports.

    All methods return HTML strings that can be embedded in reports.
    """

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not installed. Charts will be disabled.")

    def accuracy_comparison(
        self,
        metrics: Dict[str, AggregatedMetrics],
        title: str = "Accuracy Comparison Across Experiments",
    ) -> str:
        """
        Create a bar chart comparing accuracy across experiments.

        Args:
            metrics: Dict mapping experiment name to AggregatedMetrics
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        names = list(metrics.keys())
        exact_acc = [m.accuracy_exact * 100 for m in metrics.values()]
        tol_acc = [m.accuracy_tolerance * 100 for m in metrics.values()]

        fig = go.Figure(data=[
            go.Bar(name='Exact Match', x=names, y=exact_acc, marker_color='#1f77b4'),
            go.Bar(name='Within 5% Tolerance', x=names, y=tol_acc, marker_color='#2ca02c'),
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Experiment",
            yaxis_title="Accuracy (%)",
            barmode='group',
            template='plotly_white',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def cost_vs_accuracy_scatter(
        self,
        metrics: Dict[str, AggregatedMetrics],
        title: str = "Cost vs Accuracy Trade-off",
    ) -> str:
        """
        Create a scatter plot of cost vs accuracy.

        Args:
            metrics: Dict mapping experiment name to AggregatedMetrics
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        data = []
        for name, m in metrics.items():
            data.append({
                "experiment": name,
                "accuracy": m.accuracy_tolerance * 100,
                "cost": m.total_cost_usd,
                "level": m.experiment_level,
            })

        fig = go.Figure()

        for d in data:
            fig.add_trace(go.Scatter(
                x=[d["cost"]],
                y=[d["accuracy"]],
                mode='markers+text',
                name=d["experiment"],
                text=[d["experiment"]],
                textposition="top center",
                marker=dict(size=15),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Total Cost (USD)",
            yaxis_title="Accuracy (%)",
            template='plotly_white',
            height=400,
            showlegend=False,
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def per_calculator_heatmap(
        self,
        metrics: Dict[str, AggregatedMetrics],
        title: str = "Accuracy by Calculator Type",
    ) -> str:
        """
        Create a heatmap showing accuracy by calculator and experiment.

        Args:
            metrics: Dict mapping experiment name to AggregatedMetrics
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        # Get all calculators
        all_calculators = set()
        for m in metrics.values():
            all_calculators.update(m.by_calculator.keys())

        calculators = sorted(all_calculators)[:20]  # Limit to 20 for readability
        experiments = list(metrics.keys())

        # Build matrix
        z = []
        for calc in calculators:
            row = []
            for exp in experiments:
                if calc in metrics[exp].by_calculator:
                    acc = metrics[exp].by_calculator[calc].accuracy_tolerance * 100
                else:
                    acc = 0
                row.append(acc)
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=experiments,
            y=calculators,
            colorscale='RdYlGn',
            zmin=0,
            zmax=100,
            colorbar=dict(title="Accuracy (%)"),
        ))

        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600,
            xaxis_title="Experiment",
            yaxis_title="Calculator",
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def learning_curve(
        self,
        curve_data: List[Dict[str, Any]],
        experiment_name: str,
        title: str = "Learning Curve",
    ) -> str:
        """
        Create a line chart showing accuracy over time.

        Args:
            curve_data: List of {batch, accuracy} dicts
            experiment_name: Name of the experiment
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        if not curve_data:
            return "<p>No learning curve data</p>"

        batches = [d["batch"] for d in curve_data]
        accuracies = [d["accuracy"] * 100 for d in curve_data]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=batches,
            y=accuracies,
            mode='lines+markers',
            name=experiment_name,
            line=dict(width=2),
            marker=dict(size=8),
        ))

        fig.update_layout(
            title=f"{title}: {experiment_name}",
            xaxis_title="Batch",
            yaxis_title="Accuracy (%)",
            template='plotly_white',
            height=350,
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def latency_distribution(
        self,
        metrics: Dict[str, AggregatedMetrics],
        title: str = "Latency Distribution",
    ) -> str:
        """
        Create a box plot showing latency distribution across experiments.

        Args:
            metrics: Dict mapping experiment name to AggregatedMetrics
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        fig = go.Figure()

        for name, m in metrics.items():
            # Approximate distribution from summary stats
            fig.add_trace(go.Box(
                y=[m.min_latency_ms, m.p50_latency_ms, m.avg_latency_ms, m.p95_latency_ms, m.max_latency_ms],
                name=name,
                boxpoints=False,
            ))

        fig.update_layout(
            title=title,
            yaxis_title="Latency (ms)",
            template='plotly_white',
            height=400,
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def category_comparison(
        self,
        metrics: Dict[str, AggregatedMetrics],
        title: str = "Accuracy by Category",
    ) -> str:
        """
        Create a grouped bar chart comparing equation vs rule-based.

        Args:
            metrics: Dict mapping experiment name to AggregatedMetrics
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        experiments = list(metrics.keys())
        categories = ["equation", "rule"]

        fig = go.Figure()

        for cat in categories:
            accs = []
            for exp in experiments:
                cat_metrics = metrics[exp].by_category
                # Find matching category
                matching = [v for k, v in cat_metrics.items() if cat in k.lower()]
                if matching:
                    accs.append(matching[0].accuracy_tolerance * 100)
                else:
                    accs.append(0)

            fig.add_trace(go.Bar(
                name=cat.capitalize(),
                x=experiments,
                y=accs,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Experiment",
            yaxis_title="Accuracy (%)",
            barmode='group',
            template='plotly_white',
            height=400,
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def improvement_over_baseline(
        self,
        metrics: Dict[str, AggregatedMetrics],
        baseline_name: str = "baseline",
        title: str = "Improvement Over Baseline",
    ) -> str:
        """
        Create a bar chart showing improvement over baseline.

        Args:
            metrics: Dict mapping experiment name to AggregatedMetrics
            baseline_name: Name of the baseline experiment
            title: Chart title

        Returns:
            HTML string for embedding
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Plotly not available</p>"

        if baseline_name not in metrics:
            return "<p>Baseline not found</p>"

        baseline_acc = metrics[baseline_name].accuracy_tolerance

        improvements = []
        names = []
        for name, m in metrics.items():
            if name != baseline_name:
                improvement = (m.accuracy_tolerance - baseline_acc) * 100
                improvements.append(improvement)
                names.append(name)

        colors = ['#2ca02c' if i > 0 else '#d62728' for i in improvements]

        fig = go.Figure(data=go.Bar(
            x=names,
            y=improvements,
            marker_color=colors,
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Experiment",
            yaxis_title="Improvement (percentage points)",
            template='plotly_white',
            height=400,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        return fig.to_html(full_html=False, include_plotlyjs='cdn')
