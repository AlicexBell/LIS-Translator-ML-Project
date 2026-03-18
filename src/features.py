"""
Modulo condiviso per estrazione feature e augmentation.
Usato da: extractor.py (training), app.py e start.py (inferenza).

Qualsiasi modifica alle feature DEVE restare coerente tra training e inferenza.
"""

import numpy as np

# ── Nomi colonne (contratto tra training e inferenza) ─────────────────────────

# 63 coordinate raw (wrist-relative, max-abs normalized)
_RAW_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]

# 25 feature geometriche
_GEO_COLS = (
    # A) 10 distanze fingertip-to-fingertip
    ["d_4_8", "d_4_12", "d_4_16", "d_4_20",
     "d_8_12", "d_8_16", "d_8_20",
     "d_12_16", "d_12_20", "d_16_20"] +
    # B) 5 distanze fingertip-to-wrist
    ["dtw_4", "dtw_8", "dtw_12", "dtw_16", "dtw_20"] +
    # C) 5 curl ratio (tip-to-MCP / tip-to-wrist)
    ["curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky"] +
    # D) 5 angoli PIP / IP
    ["ang_thumb_ip", "ang_index_pip", "ang_middle_pip", "ang_ring_pip", "ang_pinky_pip"]
)

FEATURE_COLUMNS = _RAW_COLS + _GEO_COLS  # 88 colonne totali


# ── Helpers geometrici ────────────────────────────────────────────────────────

def _dist(a, b):
    """Distanza euclidea 3D tra due landmark MediaPipe."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _angle(a, b, c):
    """Angolo in b formato dal segmento a-b-c, in [0, pi]."""
    v1 = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    v2 = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


# ── Feature extraction principale ────────────────────────────────────────────

def normalize_and_extract(hand_landmarks):
    """
    Riceve una lista di 21 oggetti landmark MediaPipe (con attributi .x .y .z).
    Restituisce una lista di 88 float: 63 coordinate raw normalizzate + 25 geometriche.
    """
    wrist = hand_landmarks[0]

    # 1. Traslazione rispetto al polso
    coords = [
        (lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z)
        for lm in hand_landmarks
    ]

    # 2. Scalatura max-abs
    flat = [v for pt in coords for v in pt]
    max_val = max(max(abs(v) for v in flat), 1e-6)
    raw = [v / max_val for v in flat]  # 63 valori

    # 3. Feature geometriche
    # A) Distanze fingertip-to-fingertip (divise per max_val)
    pairs = [(4,8),(4,12),(4,16),(4,20),(8,12),(8,16),(8,20),(12,16),(12,20),(16,20)]
    tip_dists = [_dist(coords[a], coords[b]) / max_val for a, b in pairs]

    # B) Distanze fingertip-to-wrist (polso = origine dopo traslazione)
    tips = [4, 8, 12, 16, 20]
    tip_to_wrist = [_dist(coords[t], (0.0, 0.0, 0.0)) / max_val for t in tips]

    # C) Curl ratio: dist(tip, MCP) / dist(tip, wrist)
    tip_mcp = [(4,2),(8,5),(12,9),(16,13),(20,17)]
    curl = []
    for tip_i, mcp_i in tip_mcp:
        d_tip_wrist = _dist(coords[tip_i], (0.0, 0.0, 0.0))
        d_tip_mcp   = _dist(coords[tip_i], coords[mcp_i])
        curl.append(d_tip_mcp / max(d_tip_wrist, 1e-6))

    # D) Angoli PIP/IP (normalizzati in [0,1] dividendo per pi)
    pip_triples = [(2,3,4),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
    angles = [_angle(coords[a], coords[b], coords[c]) / np.pi for a, b, c in pip_triples]

    return raw + tip_dists + tip_to_wrist + curl + angles  # 88 valori


# ── Data augmentation (solo per training) ────────────────────────────────────

def augment_landmarks(coords_88, n_aug=5):
    """
    Riceve una lista di 88 float (già estratti da normalize_and_extract).
    Restituisce n_aug varianti augmentate come lista di liste.
    Usare SOLO durante l'estrazione del dataset, non all'inferenza.
    """
    arr = np.array(coords_88, dtype=np.float32)
    variants = []
    for _ in range(n_aug):
        v = arr.copy()
        # Rotazione 2D sul piano X-Y (prime 63 coordinate = raw landmarks)
        angle = np.random.uniform(-12, 12)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        for i in range(21):
            x = v[i*3]
            y = v[i*3 + 1]
            v[i*3]     = c*x - s*y
            v[i*3 + 1] = s*x + c*y
        # Rumore gaussiano sulle prime 63
        v[:63] += np.random.normal(0, 0.015, 63).astype(np.float32)
        # Scale jitter (30% di probabilità)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.92, 1.08)
            v[:63] *= scale
        v = np.clip(v, -2.0, 2.0)
        variants.append(v.tolist())
    return variants


def mirror_landmarks(coords_88):
    """
    Specchia orizzontalmente un campione (nega le x dei 21 landmark raw).
    Restituisce una singola lista di 88 float.
    Usare SOLO durante l'estrazione del dataset.
    """
    v = list(coords_88)
    for i in range(21):
        v[i*3] = -v[i*3]   # nega x del landmark i
    return v
