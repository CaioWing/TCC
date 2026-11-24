"""Avaliação com todos os objetos visíveis como referência, incluindo quadros omitidos."""
from collections import defaultdict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dados import CACHE, DATA, SPLITS, STRATEGIES

CLASSES = SPLITS["classes_avaliadas"]
IOUS = np.linspace(.5, .95, 10)


def ground(boxes, pose, size, gsd):
    uv = (boxes[:, :2] + boxes[:, 2:4]) / 2
    right, front = (uv[:, 0] - size[0]/2)*gsd, (size[1]/2-uv[:, 1])*gsd
    e, n, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((e + s*front + c*right, n + c*front - s*right))


def labels(video, info):
    frames = {int(f): i for i, f in enumerate(info["frames"])}
    boxes = [[] for _ in frames]
    for _, track in ET.iterparse(DATA/"ann"/f"{video}.xml", events=["end"]):
        if track.tag != "track":
            continue
        if track.get("label") in CLASSES:
            cls = CLASSES.index(track.get("label"))
            for box in track.iter("box"):
                frame = int(box.get("frame"))
                if frame in frames and box.get("outside") != "1":
                    coords = [float(box.get(a)) for a in ("xtl","ytl","xbr","ybr")]
                    boxes[frames[frame]].append([*coords, int(track.get("id")), cls])
        track.clear()
    return [np.array(b, dtype=float).reshape(-1, 6) for b in boxes]


def load_clips(video, engine, *, cache=CACHE):
    with np.load(cache/f"{video}.npz") as data:
        info = dict(data)
    gt = labels(video, info)
    with np.load(cache/engine/f"{video}.npz") as data:
        if not np.array_equal(info["frames"], data["frames"]):
            raise ValueError("As predições e as máscaras usam quadros diferentes")
        if "requested" in info and ("computed" not in data or not np.array_equal(info["requested"], data["computed"])):
            raise ValueError("Os recortes inferidos não correspondem ao plano do teste")
        det, offsets = data["det"], data["offsets"]
    predictions = []
    ntiles = len(info["tiles"])
    if len(offsets) != len(gt)*ntiles+1 or offsets[-1] != len(det):
        raise ValueError("Arquivo de predições incompleto")
    for frame in range(len(gt)):
        tiles = []
        for j, (x, y) in enumerate(info["tiles"]):
            k = frame*ntiles+j
            rec = det[offsets[k]:offsets[k+1]].copy()
            rec[:, :4] += (x,y,x,y)
            tiles.append(rec)
        predictions.append(tiles)
    cuts = np.searchsorted(info["frames"], info["starts"][1:])
    for clipno, chunk in enumerate(np.split(np.arange(len(gt)), cuts),1):
        truth = defaultdict(list)
        by_frame = [gt[i] for i in chunk]
        for i in chunk:
            xy = ground(gt[i], info["pose"][i], info["size"], info["gsd"])
            for box, point in zip(gt[i],xy):
                truth[int(box[5]),int(box[4])].append(point)
        centers = {k:np.median(points,axis=0) for k,points in truth.items()}
        scatter = [d for k,points in truth.items() for d in np.linalg.norm(points-centers[k],axis=1)]
        yield dict(video=video, clip=clipno, gt=by_frame, centers=centers, scatter=scatter,
                   pose=info["pose"][chunk], size=info["size"], gsd=info["gsd"],
                   tiles=[predictions[i] for i in chunk], masks=info["masks"][chunk],
                   frames=info["frames"][chunk])


def selected(clip, strategy):
    index = STRATEGIES.index(strategy)
    return selected_tiles(clip, clip["masks"][:,:,index])


def indices_nms_quadro(caixas):
    """Índices de caixas em coordenadas do quadro, após NMS por classe."""
    boxes = caixas[:, :4].copy()
    boxes[:, 2:] -= boxes[:, :2]
    ix = cv2.dnn.NMSBoxesBatched(boxes.tolist(), caixas[:, 4].tolist(),
                                 caixas[:, 5].astype(int).tolist(), .001, .7)
    return np.asarray(ix, dtype=int).ravel()


def selected_tiles(clip, masks):
    if len(masks) != len(clip["tiles"]):
        raise ValueError("A seleção e as previsões precisam usar os mesmos quadros.")
    frames = []
    for tiles, mask in zip(clip["tiles"], masks):
        parts = [tile for tile, use in zip(tiles,mask) if use and len(tile)]
        if not parts:
            frames.append(np.empty((0,6),np.float32))
            continue
        a = np.concatenate(parts)
        frames.append(a[indices_nms_quadro(a)])
    return frames


def pairs(a, b, radius):
    if not len(a) or not len(b):
        return []
    dist = cdist(a, b)
    penalty = (min(dist.shape)+1)*(radius+1)
    ii, jj = linear_sum_assignment(np.where(dist <= radius, dist, penalty))
    return [(i,j) for i,j in zip(ii,jj) if dist[i,j] <= radius]


def projected(clip, frames):
    result = []
    for frame,pose in zip(frames,clip["pose"]):
        frame = frame[frame[:,4]>=.15]
        points = ground(frame,pose,clip["size"],clip["gsd"])
        result.append(np.column_stack([points,frame[:,4:]]))
    return result


def associate_observations(centers, counts, xy, radius):
    """Atualiza a memória de uma classe e devolve o índice de cada observação nela."""
    indices = np.full(len(xy), -1, dtype=int)
    for i, j in pairs(centers, xy, radius):
        counts[i] += 1
        centers[i] += (xy[j] - centers[i]) / counts[i]
        indices[j] = i
    new = indices < 0
    indices[new] = len(centers) + np.arange(new.sum())
    centers = np.concatenate([centers, xy[new]])
    counts = np.r_[counts, np.ones(new.sum(), dtype=int)]
    return centers, counts, indices


def focus_assignments(clip, projections, conf, radius, dmax):
    """Centros finais e pares usados nas métricas do mapa, separados por classe."""
    for cls in range(len(CLASSES)):
        centers = np.empty((0,2))
        counts = np.zeros(0,dtype=int)
        for proj in projections:
            xy = proj[(proj[:,3]==cls)&(proj[:,2]>=conf),:2]
            if not len(xy):
                continue
            centers, counts, _ = associate_observations(centers, counts, xy, radius)
        ids = [k for k in clip["centers"] if k[0]==cls]
        assignment = pairs(centers,[clip["centers"][k] for k in ids],dmax)
        yield cls, centers, counts, ids, assignment


def focus_counts(clip, projections, conf, radius, dmax, *, return_matches=False):
    result = []
    matched = set()
    for _, centers, _, ids, assignment in focus_assignments(clip, projections, conf, radius, dmax):
        matched.update(ids[j] for _,j in assignment)
        tp = len(assignment)
        result.append([tp,len(centers)-tp,len(ids)-tp])
    counts = np.asarray(result,dtype=int)
    return (counts,matched) if return_matches else counts


def box_ious(pred, gt):
    lo = np.maximum(pred[:,None,:2],gt[None,:,:2])
    hi = np.minimum(pred[:,None,2:4],gt[None,:,2:4])
    inter = np.maximum(hi-lo,0).prod(axis=2)
    area_p = np.maximum(pred[:,2:4]-pred[:,:2],0).prod(axis=1)
    area_g = np.maximum(gt[:,2:4]-gt[:,:2],0).prod(axis=1)
    return inter/np.maximum(area_p[:,None]+area_g-inter, 1e-12)


def frame_metrics(clip, frames, conf, *, track_scores=None):
    """AP interpolada em 101 níveis de recall, por classe, com até 100 previsões/classe/quadro."""
    result = []
    for cls in range(len(CLASSES)):
        score_parts, tp_parts = [], []
        gt_count = 0
        found = set()
        for pred,gt in zip(frames,clip["gt"]):
            gt = gt[gt[:,5]==cls]
            gt_count += len(gt)
            pred = pred[pred[:,5]==cls]
            pred = pred[np.argsort(-pred[:,4],kind="stable")[:100]]
            if not len(pred):
                continue
            marks = np.zeros((len(pred),len(IOUS)),dtype=bool)
            if len(gt):
                overlaps = box_ious(pred,gt)
                used = np.zeros((len(IOUS),len(gt)),dtype=bool)
                for i in np.flatnonzero(overlaps.max(axis=1)>=.5):
                    available = np.where(used,-1.,overlaps[i])
                    best = available.argmax(axis=1)
                    ok = available[np.arange(len(IOUS)),best]>=IOUS
                    marks[i] = ok
                    used[ok,best[ok]]=True
                    if ok[0]:
                        ident = int(gt[best[0],4])
                        if track_scores is not None:
                            key = (cls, ident)
                            track_scores[key] = max(track_scores.get(key, 0), float(pred[i,4]))
                        if pred[i,4]>=conf:
                            found.add(ident)
            score_parts.append(pred[:,4])
            tp_parts.append(marks)
        scores = np.concatenate(score_parts) if score_parts else np.empty(0)
        tp = np.concatenate(tp_parts) if tp_parts else np.empty((0,len(IOUS)),bool)
        result.append((scores,tp,gt_count,len(found)))
    return result


def ap_class(records):
    scores = np.concatenate([r[0] for r in records])
    tp = np.concatenate([r[1] for r in records])
    n = sum(r[2] for r in records)
    if not n:
        return np.full(len(IOUS),np.nan)
    if not len(scores):
        return np.zeros(len(IOUS))
    order = np.argsort(-scores,kind="stable")
    cumulative = np.cumsum(tp[order],axis=0)
    precision = cumulative/np.arange(1,len(scores)+1)[:,None]
    envelope = np.maximum.accumulate(precision[::-1],axis=0)[::-1]
    ap = []
    for j in range(len(IOUS)):
        inds = np.searchsorted(cumulative[:,j]/n,np.linspace(0,1,101))
        ap.append(np.r_[envelope[:,j],0][inds].mean())
    return np.array(ap)
