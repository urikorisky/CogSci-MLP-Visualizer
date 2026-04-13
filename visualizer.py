import networkx as nx
import plotly.graph_objects as go
import numpy as np

def build_graph_layout(architecture):
    G = nx.DiGraph()
    pos = {}
    node_id_to_layer = {}
    
    current_node = 0
    layer_offsets = []
    max_nodes = max(architecture)
    
    for l_idx, num_nodes in enumerate(architecture):
        if num_nodes == 1:
            y_positions = [0.0]
        else:
            y_positions = np.linspace(1.0, -1.0, num_nodes)
            
        layer_nodes = []
        for n_idx in range(num_nodes):
            G.add_node(current_node)
            pos[current_node] = (l_idx * 2, y_positions[n_idx]) 
            node_id_to_layer[current_node] = (l_idx, n_idx)
            layer_nodes.append(current_node)
            current_node += 1
            
        layer_offsets.append(layer_nodes)
        
        if l_idx > 0:
            prev_nodes = layer_offsets[l_idx - 1]
            for u in prev_nodes:
                for v in layer_nodes:
                    G.add_edge(u, v)
                    
    return G, pos, layer_offsets, node_id_to_layer

def create_network_plot(architecture, weights, activations=None, is_delta=False, target_class_names=None):
    G, pos, layer_offsets, node_id_to_layer = build_graph_layout(architecture)
    
    min_w, max_w = -1, 1
    if weights and len(weights) > 0:
        flattened = np.concatenate([w.flatten() for w in weights])
        max_w = np.max(flattened) + 1e-5
        min_w = np.min(flattened) - 1e-5
    
    # Use 20 color bins to drastically reduce number of traces
    num_bins = 20
    traces = []
    
    bins_x = {i: [] for i in range(num_bins)}
    bins_y = {i: [] for i in range(num_bins)}
    bins_color = {}
    
    max_abs = max(abs(min_w), abs(max_w))
    
    for i in range(num_bins):
        if is_delta:
            norm_val = i / (num_bins - 1)
            r, g, b = int(norm_val * 255), int(norm_val * 200), 50
            bins_color[i] = f'rgba({r}, {g}, {b}, 0.6)'
        else:
            norm_val = i / (num_bins - 1)
            r = int((1-norm_val)*255)
            b = int(norm_val*255)
            bins_color[i] = f'rgba({r}, 50, {b}, 0.5)'
            
    for l_idx in range(len(architecture)-1):
        w_mat = weights[l_idx] if weights else np.zeros((architecture[l_idx+1], architecture[l_idx]))
        
        for out_n in range(w_mat.shape[0]):
            for in_n in range(w_mat.shape[1]):
                val = w_mat[out_n, in_n]
                u = layer_offsets[l_idx][in_n]
                v = layer_offsets[l_idx+1][out_n]
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                if is_delta:
                    max_d = max_w if max_w > 0 else 0.01
                    norm_val = np.clip(val / max_d, 0, 1)
                else:
                    norm_val = (val + max_abs) / (2 * max_abs)
                    norm_val = np.clip(norm_val, 0, 1)
                    
                bin_idx = int(norm_val * (num_bins - 1))
                bin_idx = min(bin_idx, num_bins - 1)
                bins_x[bin_idx].extend([x0, x1, None])
                bins_y[bin_idx].extend([y0, y1, None])
                
    for i in range(num_bins):
        if len(bins_x[i]) > 0:
            traces.append(go.Scatter(
                x=bins_x[i], y=bins_y[i], mode='lines',
                line=dict(width=1.5, color=bins_color[i]),
                hoverinfo='none', showlegend=False
            ))
            
    # Add colorbar for edges
    if is_delta:
        edge_cmin = 0
        edge_cmax = max_w if max_w > 0 else 0.01
        custom_colorscale = [[0, 'rgba(0,0,50,0.6)'], [1, 'rgba(255,200,50,0.6)']]
    else:
        edge_cmin = -max_abs
        edge_cmax = max_abs
        custom_colorscale = [[0, 'rgba(255,50,0,0.5)'], [0.5, 'rgba(127,50,127,0.5)'], [1, 'rgba(0,50,255,0.5)']]

    traces.append(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale=custom_colorscale,
            cmin=edge_cmin, cmax=edge_cmax,
            showscale=True,
            colorbar=dict(title="Deltas" if is_delta else "Weights", thickness=15, len=0.45, x=1.1, y=0.75)
        ),
        hoverinfo='none', showlegend=False
    ))
    
    node_x, node_y, node_colors = [], [], []
    node_text, node_textposition = [], []
    flattened_act = [0] * sum(architecture)
    if activations:
        flattened_act = []
        for act in activations:
            flattened_act.extend(act.flatten().tolist())
        
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        val = flattened_act[node] if node < len(flattened_act) else 0
        node_colors.append(val)
        
        l_idx, n_idx = node_id_to_layer[node]
        if l_idx == len(architecture) - 1 and target_class_names is not None:
            if n_idx < len(target_class_names):
                node_text.append(str(target_class_names[n_idx]))
            else:
                node_text.append("")
            node_textposition.append("middle right")
        else:
            node_text.append("")
            node_textposition.append("top center")
        
    max_c = max(node_colors) if len(node_colors) > 0 and max(node_colors) > 0 else 1
        
    traces.append(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text,
        textposition=node_textposition,
        textfont=dict(color="black", size=11, family="Arial Black"),
        marker=dict(size=15, color=node_colors, colorscale='YlGn' if not is_delta else 'Greys',
                    line=dict(width=2, color='black'), cmin=0, cmax=1 if not is_delta else max_c),
        hoverinfo='none', showlegend=False
    ))
    
    # Add colorbar for nodes
    traces.append(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale='YlGn' if not is_delta else 'Greys',
            cmin=0, cmax=1 if not is_delta else max_c,
            showscale=True,
            colorbar=dict(title="Activations", thickness=15, len=0.45, x=1.1, y=0.25)
        ),
        hoverinfo='none', showlegend=False
    ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(plot_bgcolor='white', margin=dict(l=0, r=0, b=0, t=0),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      uirevision='constant')
    return fig

def create_network_svg(architecture, weights, activations=None, is_delta=False, target_class_names=None, global_max_delta=0.0, feature_names=None):
    G, pos, layer_offsets, node_id_to_layer = build_graph_layout(architecture)
    
    width = 1100
    height = 420
    padding_x = 150
    padding_y = 20
    
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    
    min_x, max_x = min(xs) if xs else 0, max(xs) if xs else 1
    min_y, max_y = min(ys) if ys else 0, max(ys) if ys else 1
    
    def scale_x(x):
        return padding_x + (x - min_x) / (max_x - min_x) * (width - 2*padding_x) if max_x > min_x else width/2
    def scale_y(y):
        return padding_y + (max_y - y) / (max_y - min_y) * (height - 2*padding_y) if max_y > min_y else height/2

    min_w, max_w = -1, 1
    if weights and len(weights) > 0:
        flattened = np.concatenate([w.flatten() for w in weights])
        max_w = float(np.max(flattened)) + 1e-5
        min_w = float(np.min(flattened)) - 1e-5
    
    if is_delta:
        if global_max_delta > 0:
            max_w = global_max_delta
            min_w = 0.0
        max_abs = max(abs(min_w), abs(max_w))
    else:
        max_w = 1.5
        min_w = -1.5
        max_abs = 1.5
    
    svg = [f'<svg width="100%" height="auto" viewBox="0 0 {width} {height}" style="background-color: #f9f9fa; border-radius: 8px;" xmlns="http://www.w3.org/2000/svg">']
    
    svg.append('<defs>')
    if is_delta:
        svg.append('<linearGradient id="edgeGradD" x1="0%" y1="100%" x2="0%" y2="0%">')
        svg.append('  <stop offset="0%" stop-color="rgb(0,0,0)" />')
        svg.append('  <stop offset="100%" stop-color="rgb(255,255,0)" />')
        svg.append('</linearGradient>')
    else:
        svg.append('<linearGradient id="edgeGradW" x1="0%" y1="100%" x2="0%" y2="0%">')
        svg.append('  <stop offset="0%" stop-color="rgb(255,0,0)" />')
        svg.append('  <stop offset="50%" stop-color="rgb(178,178,178)" />')
        svg.append('  <stop offset="100%" stop-color="rgb(0,255,0)" />')
        svg.append('</linearGradient>')
    
    node_stops = [
        (0.0, (0, 0, 0)),
        (0.5, (0, 0, 150)),
        (0.6, (150, 0, 0)),
        (0.7, (150, 0, 150)),
        (0.8, (255, 180, 150)),
        (0.9, (255, 220, 190)),
        (1.0, (255, 255, 255))
    ]

    if not is_delta:
        svg.append('<linearGradient id="nodeGrad" x1="0%" y1="100%" x2="0%" y2="0%">')
        for pct, color in node_stops:
            svg.append(f'  <stop offset="{pct*100}%" stop-color="rgb({color[0]},{color[1]},{color[2]})" />')
        svg.append('</linearGradient>')
    svg.append('</defs>')

    for l_idx in range(len(architecture)-1):
        w_mat = weights[l_idx] if weights else np.zeros((architecture[l_idx+1], architecture[l_idx]))
        for out_n in range(w_mat.shape[0]):
            for in_n in range(w_mat.shape[1]):
                val = float(w_mat[out_n, in_n])
                u = layer_offsets[l_idx][in_n]
                v = layer_offsets[l_idx+1][out_n]
                x1, y1 = scale_x(pos[u][0]), scale_y(pos[u][1])
                x2, y2 = scale_x(pos[v][0]), scale_y(pos[v][1])
                
                if is_delta:
                    max_d = max_w if max_w > 0 else 0.01
                    norm_val = np.clip(val / max_d, 0, 1)
                    r, g, b = int(norm_val * 255), int(norm_val * 255), 0
                    color = f'rgba({r}, {g}, {b}, 0.8)'
                else:
                    norm_val = np.clip(val / max_abs, -1, 1) if max_abs > 0 else 0
                    if norm_val < 0:
                        f = (norm_val - (-1.0)) / 1.0
                        r, g, b = int(255 * (1-f) + 178 * f), int(178 * f), int(178 * f)
                    else:
                        f = norm_val / 1.0
                        r, g, b = int(178 * (1-f)), int(178 * (1-f) + 255 * f), int(178 * (1-f))
                    color = f'rgba({r}, {g}, {b}, 0.8)'
                    
                svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5" />')

    flattened_act = [0] * sum(architecture)
    if activations:
        flattened_act = []
        for act in activations:
            flattened_act.extend(act.flatten().tolist())
    
    max_c = float(max(flattened_act)) if len(flattened_act) > 0 and max(flattened_act) > 0 else 1.0

    for node in G.nodes():
        x, y = scale_x(pos[node][0]), scale_y(pos[node][1])
        val = float(flattened_act[node]) if node < len(flattened_act) else 0
        
        l_idx, n_idx = node_id_to_layer[node]
        if is_delta:
            norm_c = np.clip(val / max_c, 0, 1) if max_c > 0 else 0
            c = int(255 - norm_c * 150)
            fill = f'rgb({c}, {c}, {c})'
        else:
            if l_idx == 0:
                norm_c = np.clip(val, 0, 1)
            else:
                norm_c = np.clip(val / max_c, 0, 1) if max_c > 0 else 0
                
            r, g, b = 255, 255, 255
            for i in range(len(node_stops) - 1):
                if node_stops[i][0] <= norm_c <= node_stops[i+1][0]:
                    f = (norm_c - node_stops[i][0]) / (node_stops[i+1][0] - node_stops[i][0])
                    c1 = node_stops[i][1]
                    c2 = node_stops[i+1][1]
                    r = int(c1[0] * (1 - f) + c2[0] * f)
                    g = int(c1[1] * (1 - f) + c2[1] * f)
                    b = int(c1[2] * (1 - f) + c2[2] * f)
                    break
            fill = f'rgb({r}, {g}, {b})'
            
        svg.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{fill}" stroke="#333" stroke-width="1.5" />')
        
        if l_idx == len(architecture) - 1 and target_class_names is not None:
            if n_idx < len(target_class_names):
                label = str(target_class_names[n_idx])
                svg.append(f'<text x="{x + 18}" y="{y + 4}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" fill="#333">{label}</text>')
                
        if l_idx == 0 and feature_names is not None:
            if n_idx < len(feature_names):
                label = str(feature_names[n_idx])
                svg.append(f'<text x="{x - 18}" y="{y + 4}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333" text-anchor="end">{label}</text>')
                
    # Legends
    cb_x = width - 60
    cb_y = 30
    cb_h = 100
    cb_w = 15
    grad_id = "edgeGradD" if is_delta else "edgeGradW"
    svg.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="url(#{grad_id})" />')
    title = "Deltas" if is_delta else "Weights"
    svg.append(f'<text x="{cb_x-15}" y="{cb_y-12}" font-family="Arial" font-size="14" font-weight="bold" fill="#333">{title}</text>')
    
    if is_delta:
        top_val = f"{max_w if max_w > 0 else 0.01:.2f}"
        bot_val = "0.00"
    else:
        top_val = f"{max_abs:.2f}"
        bot_val = f"{-max_abs:.2f}"
        
    svg.append(f'<text x="{cb_x+25}" y="{cb_y+8}" font-family="Arial" font-size="12" fill="#333">{top_val}</text>')
    svg.append(f'<text x="{cb_x+25}" y="{cb_y+cb_h}" font-family="Arial" font-size="12" fill="#333">{bot_val}</text>')
    
    if not is_delta:
        ncb_x = width - 60
        ncb_y = cb_y + cb_h + 50
        ncb_h = 100
        ncb_w = 15
        svg.append(f'<rect x="{ncb_x}" y="{ncb_y}" width="{ncb_w}" height="{ncb_h}" fill="url(#nodeGrad)" />')
        svg.append(f'<text x="{ncb_x-25}" y="{ncb_y-12}" font-family="Arial" font-size="14" font-weight="bold" fill="#333">Activations</text>')
        svg.append(f'<text x="{ncb_x+25}" y="{ncb_y+8}" font-family="Arial" font-size="12" fill="#333">{max_c:.2f}</text>')
        svg.append(f'<text x="{ncb_x+25}" y="{ncb_y+ncb_h}" font-family="Arial" font-size="12" fill="#333">0.00</text>')

    svg.append('</svg>')
    return "".join(svg)
