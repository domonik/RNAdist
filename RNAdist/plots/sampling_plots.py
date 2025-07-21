import plotly.graph_objs as go
import numpy as np
import re


def empty_figure(annotation: str = None):
    fig = go.Figure()
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 50},
        font=dict(
            size=16,
        ),
        yaxis=dict(zeroline=False),
        xaxis=dict(zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

    )
    if annotation is not None:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.5,
            text=annotation,
            showarrow=False,
            font=(dict(size=28))
        )
    fig.layout.template = "plotly_white"
    fig.update_layout(
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

    )
    return fig

def distance_histo_from_matrix(distances, i, j, color:str ="#00a082"):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=np.arange(distances.shape[-1]),
            y=distances[i, j] / distances[i, j].sum(),
            marker=dict(color=color),
        )
    )
    fig.update_layout(
        xaxis=dict(title="Distance [nt]"),
        yaxis=dict(title="Probability")
    )
    return fig


def histogram_quantile(histos, q, total_counts: int, cum_counts = None, ):
    if cum_counts is None:
        cum_counts = np.cumsum(histos, axis=-1)

    rank1 = np.floor(q * (total_counts - 1)).astype(int)
    rank2 = np.ceil(q * (total_counts - 1)).astype(int)
    mask1 = cum_counts > rank1
    mask2 = cum_counts > rank2
    val1s = np.argmax(mask1, axis=-1)
    val2s = np.argmax(mask2, axis=-1)
    quantile_estimate = (val1s + val2s) / 2.0
    return quantile_estimate

def plot_distances_with_running_j(distances, i, color:str ="#00a082"):
    total_counts = distances[0, 0, 0]
    d = distances[i]
    cum_counts = np.cumsum(d, axis=-1)
    bin_values = np.arange(distances.shape[-1])

    q25 = histogram_quantile(d, q=0.25, cum_counts=cum_counts, total_counts=total_counts)
    median = histogram_quantile(d, q=0.5, cum_counts=cum_counts, total_counts=total_counts)
    q75 = histogram_quantile(d, q=0.75, cum_counts=cum_counts, total_counts=total_counts)
    weighted_sum = np.sum(d * bin_values, axis=-1)
    ed = np.divide(weighted_sum, total_counts)
    quantiles = np.concat((q25, q75[::-1]), axis=-1)
    fig = go.Figure()
    fillcolor = css_color_to_rgba(color, 0.3)
    x = np.arange(distances.shape[0])
    fig.add_traces(
        [
            go.Scatter(x=[i], y=[0], mode="markers", marker=dict(color=color), name="i"),
            go.Scatter(x=x, y=ed, line=dict(color=color), name="Mean"),
            go.Scatter(x=x, y=median, line=dict(dash="dash", color=color), name="Median"),
            go.Scatter(x=np.concat((x, x[::-1]), axis=-1), y=quantiles, line=dict(color="rgba(0,0,0,0)"), fillcolor=fillcolor, fill="toself", name="IQR")
        ]
    )
    fig.update_layout(hovermode="x")
    return fig




def expected_median_distance_maxtrix(distances, colorscale=None):
    bin_values = np.arange(distances.shape[-1])

    weighted_sum = np.sum(distances * bin_values, axis=-1)
    total_counts = distances[0, 0].sum()
    ed = np.divide(weighted_sum, total_counts, where=total_counts > 0)
    i_lower = np.tril_indices(ed.shape[0], -1)
    median = histogram_quantile(distances, q=0.5, total_counts=total_counts)
    ed[i_lower] = median[i_lower]
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=np.arange(distances.shape[-1]),
            y=np.arange(distances.shape[-1]),
            z=ed,
            colorscale=colorscale
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )
    fig.update_layout(
        xaxis=dict(title="Nucleotide"),
        yaxis=dict(title="Nucleotide"),
        coloraxis_colorbar=dict(
            title=dict(text="Distance"),
        )
    )
    return fig


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    length = len(hex_color)
    if length == 3:
        return tuple(int(hex_color[i] * 2, 16) for i in range(3))
    elif length == 4:
        return tuple(int(hex_color[i] * 2, 16) for i in range(3)) + (int(hex_color[3] * 2, 16) / 255,)
    elif length == 6:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    elif length == 8:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4)) + (int(hex_color[6:8], 16) / 255,)


def hsl_to_rgb(h, s, l):
    s /= 100
    l /= 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    elif 300 <= h < 360:
        r, g, b = c, 0, x
    else:
        raise ValueError
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# Regex patterns to match different CSS color formats
hex_pattern = re.compile(r'^#([0-9a-fA-F]{3,8})$')
rgb_pattern = re.compile(r'^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$')
rgba_pattern = re.compile(r'^rgba\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3}),\s*(\d+(\.\d+)?)\)$')
hsl_pattern = re.compile(r'^hsl\((\d{1,3}),\s*(\d{1,3})%,\s*(\d{1,3})%\)$')
hsla_pattern = re.compile(r'^hsla\((\d{1,3}),\s*(\d{1,3})%,\s*(\d{1,3})%,\s*(\d+(\.\d+)?)\)$')



def css_color_to_rgba(color, opacity):
    """
    Convert any acceptable CSS color to the same color with the specified opacity.

    :param color: A string representing the CSS color.
    :param opacity: A float representing the opacity value (between 0 and 1).
    :return: A string representing the color in RGBA format.
    """
    # Ensure opacity is a valid float between 0 and 1
    if not (0 <= opacity <= 1):
        raise ValueError("Opacity must be a float between 0 and 1.")


    match = hex_pattern.match(color)
    if match:
        rgba = hex_to_rgb(match.group(1))
        if len(rgba) == 4:
            return f'rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {opacity})'
        else:
            return f'rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {opacity})'

    match = rgb_pattern.match(color)
    if match:
        return f'rgba({match.group(1)}, {match.group(2)}, {match.group(3)}, {opacity})'

    match = rgba_pattern.match(color)
    if match:
        return f'rgba({match.group(1)}, {match.group(2)}, {match.group(3)}, {opacity})'

    match = hsl_pattern.match(color)
    if match:
        rgb = hsl_to_rgb(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

    match = hsla_pattern.match(color)
    if match:
        rgb = hsl_to_rgb(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

    raise ValueError("Invalid CSS color format.")



if __name__ == '__main__':
    from RNAdist.sampling.ed_sampling import distance_histogram
    import RNA
    seq = "AAUGCUCAGCAUGUGCUGCAGCGUAGCAGCUACGAGCAUCGUGAGC" * 3
    fc = RNA.fold_compound(seq)
    distances = distance_histogram(fc)
    fig = plot_distances_with_running_j(distances, 0)
    fig.show()
