"""
KAEL Knowledge Graph
Visualizes tau_relational as a clustered graph of session relationships.

Reads sessions from the store, clusters their embeddings using KMeans,
maps τ_relational sub-space to cluster coloring, and renders a force-directed
HTML graph using D3.js. Opens in the default browser.

Cluster meaning:
  - Each cluster = a region of conceptual/relational space KAEL has explored
  - Node size = importance_score
  - Node color = domain
  - Edge weight = cosine similarity between session embeddings
  - τ_relational vector projected onto 2D shown as a "gravity center"

The graph updates live from the session DB — run /graph at any point to
see how relational space has evolved.
"""

import json
import math
import webbrowser
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DOMAIN_COLORS = {
    "machine_learning": "#7F77DD",
    "philosophy":       "#D4537E",
    "physics":          "#1D9E75",
    "mathematics":      "#EF9F27",
    "astronomy":        "#378ADD",
    "biology":          "#639922",
    "code":             "#D85A30",
    "general":          "#888780",
}


def build_graph_data(store, model) -> dict:
    """
    Build graph nodes + edges from session store.
    Only sessions with embeddings are included.
    """
    import numpy as np

    sessions = store.get_recent_sessions(200)
    sessions = [s for s in sessions if s.session_embedding is not None]

    if len(sessions) < 2:
        return {"nodes": [], "edges": [], "clusters": [], "tau_relational_projection": None}

    # Embeddings may have inconsistent lengths if the model or config changed
    # between sessions. Find the most common length and align everything to it.
    from collections import Counter
    lengths = [len(s.session_embedding) for s in sessions]
    target_dim = Counter(lengths).most_common(1)[0][0]
    logger.info(f"Embedding dims in store: {dict(Counter(lengths))} -> using {target_dim}")

    def _align(emb, dim):
        if len(emb) >= dim:
            return emb[:dim]
        return emb + [0.0] * (dim - len(emb))

    sessions = [s for s in sessions if len(s.session_embedding) > 0]
    embs = np.array(
        [_align(s.session_embedding, target_dim) for s in sessions],
        dtype=np.float32
    )

    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embs_norm = embs / norms

    # Cluster using KMeans — number of clusters scales with session count
    n_clusters = max(2, min(8, len(sessions) // 5))
    labels = _kmeans(embs_norm, n_clusters)

    # Project embeddings to 2D using PCA for layout
    coords_2d = _pca_2d(embs_norm)

    # τ_relational projected to 2D (shows where identity's relational weight points)
    tau_rel = model.tau.tau_relational.detach().cpu().float().numpy()
    tau_emb = np.zeros(embs.shape[1])
    # Align tau_relational dim to embedding dim
    min_d = min(len(tau_rel), len(tau_emb))
    tau_emb[:min_d] = tau_rel[:min_d]
    tau_emb_norm = tau_emb / (np.linalg.norm(tau_emb) + 1e-8)
    tau_2d = _project_point(tau_emb_norm, embs_norm, coords_2d)

    # Build nodes
    nodes = []
    for i, s in enumerate(sessions):
        nodes.append({
            "id": s.session_id[:8],
            "full_id": s.session_id,
            "label": _truncate(s.user_input, 40),
            "domain": s.domain or "general",
            "color": DOMAIN_COLORS.get(s.domain or "general", "#888780"),
            "importance": s.importance_score or 0.3,
            "novelty": s.novelty_score or 0.5,
            "gate": s.gate_value,
            "flagged": (s.gate_value is not None and s.gate_value < 0.2),
            "cluster": int(labels[i]),
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "ts": s.timestamp,
            "response_preview": _truncate(s.model_output, 100),
        })

    # Build edges — connect sessions with high cosine similarity
    edges = []
    sim_matrix = embs_norm @ embs_norm.T
    threshold = 0.6
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                edges.append({
                    "source": sessions[i].session_id[:8],
                    "target": sessions[j].session_id[:8],
                    "weight": round(sim, 3),
                })

    # Cluster centroids for labeling
    clusters = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cluster_nodes = [sessions[i] for i in range(len(sessions)) if labels[i] == c]
        # Most common domain in cluster
        domains = [s.domain or "general" for s in cluster_nodes]
        dominant = max(set(domains), key=domains.count)
        centroid_2d = coords_2d[mask].mean(axis=0)
        clusters.append({
            "id": c,
            "label": f"{dominant} #{c}",
            "dominant_domain": dominant,
            "color": DOMAIN_COLORS.get(dominant, "#888780"),
            "size": int(mask.sum()),
            "cx": float(centroid_2d[0]),
            "cy": float(centroid_2d[1]),
        })

    # τ stats for display
    tau_stats = {
        "norm": round(float(model.tau.norm()), 4),
        "epistemic_norm": round(float(model.tau.tau_epistemic.detach().norm().item()), 4),
        "dispositional_norm": round(float(model.tau.tau_dispositional.detach().norm().item()), 4),
        "relational_norm": round(float(model.tau.tau_relational.detach().norm().item()), 4),
        "phase": model._phase,
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
        "tau_relational_projection": {"x": float(tau_2d[0]), "y": float(tau_2d[1])},
        "tau_stats": tau_stats,
        "total_sessions": store.count_sessions(),
    }


def _kmeans(X, k: int, max_iter: int = 100):
    """Minimal numpy KMeans — no sklearn dependency."""
    import numpy as np
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        dists = np.dot(X, centroids.T)  # cosine sim (X already normalized)
        new_labels = np.argmax(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = X[mask].mean(axis=0)
                n = np.linalg.norm(centroids[c])
                if n > 0:
                    centroids[c] /= n
    return labels


def _pca_2d(X):
    """Minimal numpy PCA to 2D."""
    import numpy as np
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / len(X)
    # Power iteration for top 2 eigenvectors
    v1 = _power_iter(cov)
    # Deflate
    cov2 = cov - (cov @ v1[:, None]) @ v1[None, :] * (v1 @ cov @ v1)
    v2 = _power_iter(cov2)
    coords = X_centered @ np.stack([v1, v2], axis=1)
    # Normalize to [-1, 1]
    for i in range(2):
        rng = coords[:, i].max() - coords[:, i].min()
        if rng > 0:
            coords[:, i] = (coords[:, i] - coords[:, i].min()) / rng * 2 - 1
    return coords


def _power_iter(M, n_iter: int = 50):
    import numpy as np
    v = np.random.default_rng(0).normal(size=M.shape[0])
    for _ in range(n_iter):
        v = M @ v
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
    return v


def _project_point(point, ref_matrix, coords_2d):
    """Project a single point into 2D using the same basis as ref_matrix."""
    import numpy as np
    sims = ref_matrix @ point
    weights = np.maximum(sims, 0)
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return np.array([0.0, 0.0])
    return (weights[:, None] * coords_2d).sum(axis=0) / w_sum


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    text = text.replace('"', "'").replace('\n', ' ').strip()
    return text[:n] + "..." if len(text) > n else text


def render_html(graph_data: dict) -> str:
    # Build via simple string replacement — no f-string so JS/CSS braces are safe
    data_json = json.dumps(graph_data, ensure_ascii=False)
    domain_colors_json = json.dumps(DOMAIN_COLORS)

    html = (
        _HTML_TEMPLATE
        .replace("__DATA_JSON__", data_json)
        .replace("__DOMAIN_COLORS_JSON__", domain_colors_json)
    )
    return html


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>KAEL knowledge graph</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #0e0e10; color: #c2c0b6; }
#header { padding: 16px 20px; border-bottom: 0.5px solid #2c2c2a; display: flex; align-items: center; gap: 20px; }
#header h1 { font-size: 14px; font-weight: 500; color: #f0eee8; }
.stat { font-size: 12px; color: #888780; }
.stat span { color: #AFA9EC; font-weight: 500; }
#main { display: flex; height: calc(100vh - 53px); }
#graph-container { flex: 1; position: relative; }
svg { width: 100%; height: 100%; }
#sidebar { width: 280px; border-left: 0.5px solid #2c2c2a; padding: 16px; overflow-y: auto; font-size: 12px; flex-shrink: 0; }
#sidebar h2 { font-size: 12px; font-weight: 500; color: #888780; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }
#node-detail { display: none; }
#node-detail.visible { display: block; }
.detail-label { color: #888780; margin-top: 8px; }
.detail-value { color: #f0eee8; margin-top: 2px; line-height: 1.5; }
.detail-value.mono { font-family: monospace; font-size: 11px; color: #9FE1CB; }
.cluster-pill { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; margin: 2px; }
.flagged-badge { background: #A32D2D; color: #F7C1C1; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
.legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
#empty-msg { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); text-align: center; color: #888780; }
</style>
</head>
<body>
<div id="header">
  <h1>KAEL — tau_relational knowledge graph</h1>
  <div class="stat">sessions <span id="stat-sessions">0</span></div>
  <div class="stat">tau norm <span id="stat-norm">—</span></div>
  <div class="stat">relational norm <span id="stat-rel">—</span></div>
  <div class="stat">phase <span id="stat-phase">0</span></div>
</div>
<div id="main">
  <div id="graph-container"><svg id="svg"></svg>
    <div id="empty-msg" style="display:none">
      <div style="font-size:14px;color:#534AB7;margin-bottom:8px">No sessions with embeddings yet</div>
      <div>Have a few conversations with KAEL first, then run /graph again</div>
    </div>
  </div>
  <div id="sidebar">
    <h2>Clusters</h2><div id="cluster-list"></div>
    <h2 style="margin-top:16px">Domain legend</h2><div id="legend"></div>
    <div id="node-detail" style="margin-top:16px">
      <h2>Selected session</h2>
      <div class="detail-label">input</div><div class="detail-value" id="d-input"></div>
      <div class="detail-label">response preview</div><div class="detail-value" id="d-response"></div>
      <div class="detail-label">domain</div><div class="detail-value mono" id="d-domain"></div>
      <div class="detail-label">novelty / gate / importance</div><div class="detail-value mono" id="d-scores"></div>
      <div class="detail-label">cluster</div><div class="detail-value mono" id="d-cluster"></div>
    </div>
    <div style="margin-top:20px">
      <h2>tau_relational</h2>
      <div class="detail-label">Purple cross = where tau_relational points. Sessions near it shaped KAEL's relational identity most.</div>
    </div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const DATA = __DATA_JSON__;
const DOMAIN_COLORS = __DOMAIN_COLORS_JSON__;
const nodes = DATA.nodes;
const edges = DATA.edges;
const clusters = DATA.clusters;
const tau_proj = DATA.tau_relational_projection;
const tau_stats = DATA.tau_stats || {};

document.getElementById('stat-sessions').textContent = DATA.total_sessions || nodes.length;
document.getElementById('stat-norm').textContent = tau_stats.norm || '—';
document.getElementById('stat-rel').textContent = tau_stats.relational_norm || '—';
document.getElementById('stat-phase').textContent = tau_stats.phase || 0;

const svg = d3.select('#svg');
const container = document.getElementById('graph-container');
if (nodes.length < 2) { document.getElementById('empty-msg').style.display = 'block'; }

const zoom = d3.zoom().scaleExtent([0.2,8]).on('zoom', (e) => g.attr('transform', e.transform));
svg.call(zoom);
const g = svg.append('g');
const pad = 80;
const xScale = d3.scaleLinear().domain([-1,1]).range([pad, container.clientWidth - pad]);
const yScale = d3.scaleLinear().domain([-1,1]).range([pad, container.clientHeight - pad]);
const nodeR = d => 5 + (d.importance || 0.3) * 12;
const nodeX = d => xScale(d.x);
const nodeY = d => yScale(d.y);

const halo = g.append('g');
clusters.forEach(c => {
  const cx = xScale(c.cx), cy = yScale(c.cy);
  halo.append('circle').attr('cx',cx).attr('cy',cy).attr('r',60+c.size*8)
    .attr('fill',c.color).attr('opacity',0.04).attr('stroke',c.color)
    .attr('stroke-width',0.5).attr('stroke-opacity',0.2);
  halo.append('text').attr('x',cx).attr('y',cy-65-c.size*8)
    .attr('text-anchor','middle').attr('fill',c.color).attr('font-size',11).attr('opacity',0.6).text(c.label);
});

g.append('g').selectAll('line').data(edges).join('line')
  .attr('x1', d => { const n = nodes.find(n=>n.id===d.source); return n?nodeX(n):0; })
  .attr('y1', d => { const n = nodes.find(n=>n.id===d.source); return n?nodeY(n):0; })
  .attr('x2', d => { const n = nodes.find(n=>n.id===d.target); return n?nodeX(n):0; })
  .attr('y2', d => { const n = nodes.find(n=>n.id===d.target); return n?nodeY(n):0; })
  .attr('stroke','#3C3489').attr('stroke-width', d => d.weight*2).attr('opacity',0.3);

const nodeEls = g.append('g').selectAll('g').data(nodes).join('g')
  .attr('transform', d => `translate(${nodeX(d)},${nodeY(d)})`)
  .attr('cursor','pointer')
  .on('click', (e,d) => showDetail(d))
  .on('mouseenter', function(e,d){ d3.select(this).select('circle').attr('stroke-width',2).attr('stroke','#fff'); })
  .on('mouseleave', function(e,d){ d3.select(this).select('circle').attr('stroke-width',d.flagged?1.5:0.5).attr('stroke',d.flagged?'#E24B4A':d.color); });

nodeEls.append('circle')
  .attr('r', nodeR).attr('fill', d=>d.color)
  .attr('fill-opacity', d => 0.3+(d.importance||0.3)*0.5)
  .attr('stroke', d=>d.flagged?'#E24B4A':d.color)
  .attr('stroke-width', d=>d.flagged?1.5:0.5);
nodeEls.append('title').text(d=>d.label);
nodeEls.filter(d=>d.flagged).append('circle').attr('r',3).attr('fill','#E24B4A').attr('cy', d=>-nodeR(d)-3);

if (tau_proj) {
  const tx=xScale(tau_proj.x), ty=yScale(tau_proj.y);
  const tg=g.append('g').attr('transform',`translate(${tx},${ty})`);
  tg.append('line').attr('x1',-12).attr('x2',12).attr('y1',0).attr('y2',0).attr('stroke','#534AB7').attr('stroke-width',2);
  tg.append('line').attr('x1',0).attr('x2',0).attr('y1',-12).attr('y2',12).attr('stroke','#534AB7').attr('stroke-width',2);
  tg.append('circle').attr('r',5).attr('fill','none').attr('stroke','#7F77DD').attr('stroke-width',1.5);
  tg.append('text').text('tau_relational').attr('y',-16).attr('text-anchor','middle').attr('fill','#7F77DD').attr('font-size',10);
}

document.getElementById('cluster-list').innerHTML = clusters.map(c =>
  `<span class="cluster-pill" style="background:${c.color}33;color:${c.color};border:0.5px solid ${c.color}66">${c.label} (${c.size})</span>`
).join('');

document.getElementById('legend').innerHTML = Object.entries(DOMAIN_COLORS).map(([d,c]) =>
  `<div class="legend-item"><div class="legend-dot" style="background:${c}"></div><span style="font-size:11px">${d}</span></div>`
).join('');

function showDetail(d) {
  document.getElementById('node-detail').classList.add('visible');
  document.getElementById('d-input').textContent = d.label;
  document.getElementById('d-response').textContent = d.response_preview || '—';
  document.getElementById('d-domain').textContent = d.domain;
  document.getElementById('d-scores').textContent =
    `novelty=${d.novelty?.toFixed(3)||'n/a'}  gate=${d.gate?.toFixed(3)||'n/a'}  importance=${d.importance?.toFixed(3)||'n/a'}`;
  document.getElementById('d-cluster').textContent = `cluster ${d.cluster}`;
  if (d.flagged) document.getElementById('d-cluster').innerHTML += ' <span class="flagged-badge">FLAGGED</span>';
}
</script></body></html>"""

def build_and_open_graph(store, model, output_path: Optional[Path] = None):
    """Build the graph and open it in the default browser."""
    logger.info("Building tau_relational knowledge graph...")

    try:
        graph_data = build_graph_data(store, model)
    except ImportError:
        logger.error("numpy required for knowledge graph. Run: pip install numpy")
        print("numpy required: pip install numpy")
        return

    html = render_html(graph_data)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", prefix="kael_graph_", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write(html)
        tmp.close()
        output_path = Path(tmp.name)
    else:
        output_path.write_text(html, encoding="utf-8")

    n_nodes = len(graph_data["nodes"])
    n_edges = len(graph_data["edges"])
    logger.info(f"Graph: {n_nodes} nodes, {n_edges} edges → {output_path}")
    print(f"Graph: {n_nodes} sessions, {n_edges} connections → {output_path}")

    webbrowser.open(f"file://{output_path}")
    return output_path