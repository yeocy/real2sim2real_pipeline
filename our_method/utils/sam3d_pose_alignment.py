"""
sam3d_pose_alignment.py — SAM3D 에셋의 yaw(z_angle)와 바닥면 위치를 입력 이미지에 맞춘다 (render-and-compare).

Step 2 는 풀 렌더 100장(다른 앙각, 검은 배경)과 크롭을 DINO/GPT 로 비교해 yaw 를 고른다. 원통형처럼
yaw 단서가 손잡이 하나뿐인 물체는 여기서 자주 틀린다 (artifacts/05_SAM3D_ASSET_ALIGNMENT.md 3.2).
여기서는 물체 종류에 대한 규칙 없이, 같은 에셋(실치수가 구워진 텍스처 메시)을 **입력 카메라 그대로**
렌더해서 입력 관측과 직접 비교한다.

  변수:  yaw, 바닥 중심 xy (바닥 z 는 마스크 바깥 링으로 잰 받침면에 고정, 스케일은 바꾸지 않는다)
  렌더:  open3d 레이캐스팅 (CPU, 마스크 주변 크롭만). 렌더 마스크, 깊이, 텍스처 색을 얻는다.
  비용:  (1 - 실루엣 IoU) + 깊이 차 + 색 차   (렌더가 관측보다 뒤에 있는 픽셀 = 다른 물체에 가려진 곳은 뺀다)
  탐색:  yaw 72개(5도)마다 xy 를 "보이는 면 중심 맞추기" 로 정한 뒤 비용 계산 ->
         서로 다른 극소(>= 25도 차) 상위 몇 개에서 Nelder-Mead 로 (yaw, x, y) 미세 조정.
  모호성: 비용 곡선의 전체 폭이 작으면 yaw 와 무관한 물체(회전 대칭)로 본다. 극소 두 개의 비용이
         가까우면 ambiguous 로 표시한다. 호출 측(빌더)이 그때 VLM 검증을 붙일 수 있다 (candidates 를 남긴다).

결과 pose 는 에셋 원점(= 메시 AABB 중심, build_og_usd.build_usd 가 그 점을 원점으로 둔다) 기준이다.
  pose_world: 입력 camera_info 의 월드 프레임
  pose_cam:   입력 카메라(opengl, -z 전방 +y 위) 기준. Step 3 은 T_og_cam @ pose_cam 으로 바로 놓는다.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# sam3d_assets.json 의 registration.method. 바뀌면 캐시된 풀도 align 부터 다시 돈다 (sam3d_asset_generation)
ALIGN_METHOD = "yaw_render_compare+scale_xyz_depth_registration"
YAW_STEPS = 72
MODE_SEP_DEG = 25.0
DEPTH_CLIP = 0.02          # 깊이 차 상한 [m]
OCCLUDE_EPS = 0.01         # 관측이 렌더보다 이만큼 앞이면 가려진 픽셀로 본다
W_SIL, W_DEPTH, W_COLOR = 1.0, 0.5, 0.5
# 크기: depth 점군 3D 정합으로 메시 자기 축(가로 x, 세로 y, 높이 z)별 배율을 맞춘다.
#   비용 = 관측 점 -> 렌더 표면 점 + 렌더 표면 점 -> 관측 점 (최근접 거리, PC_CLIP 에서 자름, 양방향)
#   prior = W_ASPECT * (축 사이 log 비율 - SAM3D 원래 비율)^2. 한 시점에서 거의 안 보이는 축만 붙잡는 약한 사전
#   배율이 SCALE_LIMIT 를 넘으면 큰 벌점 (mesh 단계 실측에서 크게 벗어나지 않게)
W_ASPECT = 0.2
PC_CLIP = 0.015
PC_SAMPLES = 4000
YAW_SLACK = np.radians(20.0)   # 스케일 정합 중 yaw 는 후보에서 이만큼만 움직인다 (yaw 는 전수 탐색에서 정한다)
Z_SLACK = 0.02                 # 바닥 높이는 받침면 추정값 +-2cm 안에서 맞춘다
SCALE_LIMIT = np.log(1.5)
SYMMETRIC_RANGE = 0.03     # 비용 곡선 (max - min) 이 이보다 작으면 yaw 무관 (회전 대칭)
AMBIGUOUS_GAP = 0.02       # 1, 2 위 극소의 비용 차가 이보다 작으면 모호


def _cam(info):
    K = np.array(info["intrinsics"]["matrix"], dtype=np.float64)
    Rwc = Rotation.from_quat(info["camera"]["orientation"]).as_matrix()
    pc = np.array(info["camera"]["position"], dtype=np.float64)
    return K, Rwc, pc


def backproject(depth, info, mask):
    """마스크 픽셀 -> 월드 점 (sam3d_asset_builder.backproject 와 같은 규약)."""
    K, Rwc, p = _cam(info)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], -(v - K[1, 2]) * z / K[1, 1], -z], axis=1)
    return pc @ Rwc.T + p


def pose_mat(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class CropRenderer:
    """마스크 주변 크롭에서 입력 카메라로 메시를 레이캐스팅한다. 메시는 로컬 프레임 그대로 두고 광선을 옮긴다."""

    def __init__(self, V, F, uvs, tex, depth, info, rgb, mask, pad=1.0, stride=1):
        import open3d as o3d
        self.V, self.F = V, F
        self.uvs = uvs if uvs is not None and len(uvs) == len(V) else None
        self.tex = tex
        H, W = depth.shape
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        px, py = int((x1 - x0) * pad) + 8, int((y1 - y0) * pad) + 8
        self.box = (max(x0 - px, 0), min(x1 + px + 1, W), max(y0 - py, 0), min(y1 + py + 1, H))
        bx0, bx1, by0, by1 = self.box
        vv, uu = np.mgrid[by0:by1:stride, bx0:bx1:stride]
        self.shape = vv.shape
        self.u, self.v = uu.ravel(), vv.ravel()
        K, Rwc, pc = _cam(info)
        d_cam = np.stack([(self.u - K[0, 2]) / K[0, 0], -(self.v - K[1, 2]) / K[1, 1], -np.ones(len(self.u))], 1)
        self.fwd_len = 1.0 / np.linalg.norm(d_cam, axis=1)       # 단위 광선 1m 당 planar depth
        d = d_cam * self.fwd_len[:, None]
        self.dw = d @ Rwc.T
        self.cam_pos = pc
        self.obs_mask = mask[self.v, self.u].astype(bool)
        self.obs_depth = depth[self.v, self.u]
        self.obs_rgb = rgb[self.v, self.u].astype(np.float64) / 255.0
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        self.scene = scene
        self._o3d = o3d

    def render(self, yaw, t, color=True, scale=None):
        """scale (3,): 메시 로컬 축별 배율 (바닥 z=0 기준이라 바닥은 그대로 받침면에 붙는다).

        메시는 그대로 두고 광선을 로컬로 옮긴다: o_l = S^-1 R^T (c - t), d_l = S^-1 R^T d.
        레이캐스팅의 t_hit 는 방향 벡터 배수(매개변수)라 월드 단위 광선(|d|=1)의 거리와 같다.
        """
        R = Rotation.from_euler("z", yaw).as_matrix()
        k = np.ones(3) if scale is None else np.asarray(scale, dtype=np.float64)
        o = ((self.cam_pos - t) @ R) / k            # S^-1 R^T (c - t)
        dl = (self.dw @ R) / k
        rays = np.concatenate([np.repeat(o[None], len(dl), 0), dl], 1).astype(np.float32)
        ans = self.scene.cast_rays(self._o3d.core.Tensor(rays))
        th = ans["t_hit"].numpy()
        hit = np.isfinite(th)
        n = (ans["primitive_normals"].numpy() / k) @ R.T
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        out = {"hit": hit, "depth": np.where(hit, th * self.fwd_len, np.inf),
               "pid": ans["primitive_ids"].numpy(), "bc": ans["primitive_uvs"].numpy(),
               "shade": np.abs((n * self.dw).sum(1))}
        if color:
            out["rgb"] = self.colors(out)
        out["points"] = self.cam_pos + self.dw * np.where(hit, th, 0)[:, None]
        return out

    def colors(self, r):
        hit = r["hit"]
        col = np.zeros((len(hit), 3))
        if self.uvs is None or self.tex is None:
            return col
        f = self.F[r["pid"][hit]]
        a, b = r["bc"][hit, 0], r["bc"][hit, 1]
        uv = (1 - a - b)[:, None] * self.uvs[f[:, 0]] + a[:, None] * self.uvs[f[:, 1]] + b[:, None] * self.uvs[f[:, 2]]
        S = self.tex.shape[0]
        x = np.clip((uv[:, 0] * S).astype(int), 0, S - 1)
        y = np.clip(((1 - uv[:, 1]) * S).astype(int), 0, S - 1)
        col[hit] = self.tex[y, x] / 255.0
        return col

    def score(self, r):
        """(비용, 세부). 렌더가 관측보다 뒤인데 관측 마스크 밖인 픽셀은 가림으로 보고 뺀다."""
        R, M = r["hit"], self.obs_mask
        occluded = R & ~M & (self.obs_depth < r["depth"] - OCCLUDE_EPS)
        Rv = R & ~occluded
        inter = Rv & M
        union = Rv | M
        iou = inter.sum() / max(union.sum(), 1)
        if inter.sum() > 10:
            dd = np.minimum(np.abs(r["depth"][inter] - self.obs_depth[inter]), DEPTH_CLIP) / DEPTH_CLIP
            dterm = float(dd.mean())
            cterm = float(np.minimum(np.abs(r["rgb"][inter] - self.obs_rgb[inter]).mean(1), 0.5).mean() / 0.5) \
                if "rgb" in r else 0.0
        else:
            dterm, cterm = 1.0, 1.0
        cost = W_SIL * (1 - iou) + W_DEPTH * dterm + W_COLOR * cterm
        return float(cost), {"iou": float(iou), "depth": dterm, "color": cterm, "occluded_px": int(occluded.sum())}

    def center_xy(self, yaw, t, iters=3, scale=None):
        """보이는 면 중심을 관측 점 중심에 맞춘다 (둘 다 카메라 쪽 면이라 치우침이 상쇄된다).

        바닥 높이(z)도 같이 맞추되 받침면 추정값 +-Z_SLACK 안으로 자른다 (받침면 추정 오차, 물체가 살짝
        떠 있거나 받침이 기울어진 경우를 흡수한다. 이게 없으면 깊이 차를 스케일과 yaw 가 대신 메운다).
        """
        P = self.obs_points_mean
        t = np.array(t, dtype=np.float64)
        for _ in range(iters):
            r = self.render(yaw, t, color=False, scale=scale)
            vis = r["hit"] & self.obs_mask
            if vis.sum() < 10:
                vis = r["hit"]
            if vis.sum() < 10:
                break
            delta = P - r["points"][vis].mean(0)
            t += delta
            t[2] = np.clip(t[2], *self.z_range)
            if np.linalg.norm(delta) < 5e-4:
                break
        return t


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _modes(yaws, costs, sep_deg=MODE_SEP_DEG, k=4):
    """원형 비용 곡선의 극소를 비용 순으로, 서로 sep_deg 이상 떨어진 것만."""
    n = len(costs)
    idx = [i for i in range(n) if costs[i] <= costs[(i - 1) % n] and costs[i] <= costs[(i + 1) % n]]
    idx.sort(key=lambda i: costs[i])
    out = []
    for i in idx:
        if all(abs(np.degrees(_wrap(yaws[i] - yaws[j]))) >= sep_deg for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return out


def align_yaw(V, F, uvs, tex, depth, info, rgb, mask, yaw_steps=YAW_STEPS, refine_top=3, log=print):
    """메시(로컬: xy 중심, 바닥 z=0, 실치수)의 yaw 와 바닥 중심 xy 를 입력 관측에 맞춘다.

    Args:
        depth (H,W) planar depth [m], info: camera_info (opengl), rgb (H,W,3) uint8 RGB, mask (H,W) bool
            (모두 같은 해상도)

    Returns:
        dict: yaw, t (바닥 중심 월드 위치), cost, terms, curve(yaw 별 비용), candidates(상위 극소),
              symmetric, ambiguous, support_z
    """
    from scipy.optimize import minimize

    m = mask.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    inner = cv2.erode(m, k3, iterations=2).astype(bool)
    ring = (cv2.dilate(m, k3, iterations=8) - cv2.dilate(m, k3, iterations=3)).astype(bool)
    P = backproject(depth, info, inner if inner.sum() > 30 else m.astype(bool))
    support_z = float(np.median(backproject(depth, info, ring)[:, 2]))

    rd = CropRenderer(V, F, uvs, tex, depth, info, rgb, m.astype(bool))
    rd.support_z = support_z
    rd.z_range = (support_z - Z_SLACK, support_z + Z_SLACK)
    rd.obs_P_full = P                       # 마스크 안쪽(경계 2px 제외) depth 점군, 크기 정합에 쓴다
    # 관측 중심은 렌더와 같은 크롭/마스크 픽셀에서 잰다
    obs_pts = rd.cam_pos + rd.dw * (rd.obs_depth / rd.fwd_len)[:, None]
    rd.obs_points_mean = obs_pts[rd.obs_mask].mean(0) if rd.obs_mask.sum() else P.mean(0)

    t0 = np.array([P[:, 0].mean(), P[:, 1].mean(), support_z])
    yaws = np.linspace(-np.pi, np.pi, yaw_steps, endpoint=False)
    curve, ts, terms = [], [], []
    for yaw in yaws:
        t = rd.center_xy(yaw, t0)
        c, tm = rd.score(rd.render(yaw, t))
        curve.append(c)
        ts.append(t)
        terms.append(tm)
    curve = np.array(curve)

    cands = []
    for i in _modes(yaws, curve, k=refine_top):
        tz = ts[i][2]

        def f(x):
            return rd.score(rd.render(x[0], np.array([x[1], x[2], tz])))[0]

        x0 = np.array([yaws[i], ts[i][0], ts[i][1]])
        res = minimize(f, x0, method="Nelder-Mead",
                       options={"initial_simplex": x0 + np.array([[0, 0, 0], [np.radians(4), 0, 0],
                                                                  [0, 0.006, 0], [0, 0, 0.006]]),
                                "xatol": 1e-4, "fatol": 1e-4, "maxiter": 200})
        x = res.x if res.fun <= curve[i] else x0
        t = np.array([x[1], x[2], tz])
        c, tm = rd.score(rd.render(x[0], t))
        cands.append({"yaw": float(_wrap(x[0])), "t": t.tolist(), "cost": c, "terms": tm,
                      "grid_yaw": float(yaws[i]), "grid_cost": float(curve[i])})
    cands.sort(key=lambda d: d["cost"])
    best = cands[0]
    rng = float(curve.max() - curve.min())
    symmetric = rng < SYMMETRIC_RANGE
    ambiguous = (not symmetric) and len(cands) > 1 and (cands[1]["cost"] - best["cost"]) < AMBIGUOUS_GAP
    log(f"    yaw {np.degrees(best['yaw']):+7.1f}°  cost {best['cost']:.3f} (IoU {best['terms']['iou']:.3f}, "
        f"depth {best['terms']['depth']:.3f}, color {best['terms']['color']:.3f})  "
        f"curve range {rng:.3f}{'  [symmetric]' if symmetric else ''}{'  [ambiguous]' if ambiguous else ''}  "
        f"candidates " + ", ".join(f"{np.degrees(c['yaw']):+.0f}°:{c['cost']:.3f}" for c in cands))
    return {"yaw": best["yaw"], "t": best["t"], "cost": best["cost"], "terms": best["terms"],
            "support_z": support_z, "symmetric": bool(symmetric), "ambiguous": bool(ambiguous),
            "candidates": cands, "curve": {"yaw": yaws.tolist(), "cost": curve.tolist()}, "_renderer": rd}


def _pc_terms(rd, r):
    """depth 점군 3D 정합 항: (관측 -> 렌더 표면, 렌더 표면 -> 관측) 평균 최근접 거리 / PC_CLIP.

    렌더 표면 점은 입력 카메라에서 보이는 면만 (레이캐스팅). 다른 물체에 가려진 렌더 점(관측이 1cm 이상 앞,
    마스크 밖)은 뺀다. 관측에 없는 곳으로 튀어나온 메시(너무 큼)와 관측을 못 덮는 메시(너무 작음)를 둘 다 잡는다.
    """
    from scipy.spatial import cKDTree
    occluded = r["hit"] & ~rd.obs_mask & (rd.obs_depth < r["depth"] - OCCLUDE_EPS)
    Q = r["points"][r["hit"] & ~occluded]
    if len(Q) < 20:
        return 1.0, 1.0
    if len(Q) > PC_SAMPLES:
        Q = Q[np.random.default_rng(0).choice(len(Q), PC_SAMPLES, replace=False)]
    d_pq = cKDTree(Q).query(rd.obs_P)[0]
    d_qp = rd.obs_tree.query(Q)[0]
    return (float(np.minimum(d_pq, PC_CLIP).mean() / PC_CLIP),
            float(np.minimum(d_qp, PC_CLIP).mean() / PC_CLIP))


def refine_scale_depth(rd, yaw, t, native_log=None, w_aspect=W_ASPECT):
    """yaw 후보 하나에서 (yaw, log sx, log sy, log sz, x, y, 바닥 z) 를 depth 점군 3D 정합으로 맞춘다.

    배율은 메시 자기 축(가로 x, 세로 y, 높이 z)마다 따로다. yaw 는 후보 +-YAW_SLACK(yaw 는 전수 탐색이 정한다),
    바닥 z 는 받침면 +-Z_SLACK, 배율은 +-SCALE_LIMIT 안으로 묶는다.

    Args:
        native_log (3,): SAM3D 원래 비율로 되돌리는 log 배율 (= -log(mesh 단계 배율)). 축 사이 비율 사전에 쓴다.

    Returns:
        dict: yaw, t, scale(3,), cost, pc(관측->렌더, 렌더->관측), render_cost, terms(실루엣/깊이/색, 참고용)
    """
    from scipy.optimize import minimize

    yaw0 = float(yaw)
    nl = np.zeros(3) if native_log is None else np.asarray(native_log, dtype=np.float64)

    def unpack(x):
        return x[0], np.array([x[4], x[5], x[6]]), np.exp(x[1:4])

    def f(x):
        yaw_, t_, k = unpack(x)
        a, b = _pc_terms(rd, rd.render(yaw_, t_, scale=k, color=False))
        l = x[1:4] - nl                                   # SAM3D 원래 비율 기준 log 배율
        prior = w_aspect * ((l[0] - l[1]) ** 2 + (l[2] - 0.5 * (l[0] + l[1])) ** 2)
        over = sum(max(abs(v) - SCALE_LIMIT, 0.0) for v in x[1:4])
        over += max(abs(_wrap(x[0] - yaw0)) - YAW_SLACK, 0.0)
        over += (max(rd.z_range[0] - x[6], 0.0) + max(x[6] - rd.z_range[1], 0.0)) * 20.0
        return a + b + prior + 10.0 * over

    x0 = np.array([yaw, 0.0, 0.0, 0.0, t[0], t[1], np.clip(t[2], *rd.z_range)])
    simplex = x0 + np.vstack([np.zeros(7), np.diag([np.radians(5), 0.08, 0.08, 0.08, 0.006, 0.006, 0.004])])
    best = None
    for it in range(2):                       # 재시작 한 번 (Nelder-Mead 가 일찍 멈추는 것 방지)
        res = minimize(f, x0, method="Nelder-Mead",
                       options={"initial_simplex": simplex if it == 0 else None, "xatol": 1e-4,
                                "fatol": 1e-5, "maxiter": 1200})
        if best is None or res.fun < best.fun:
            best = res
        x0 = best.x
    yaw_, t_, k = unpack(best.x)
    r = rd.render(yaw_, t_, scale=k)
    c, tm = rd.score(r)
    pc = _pc_terms(rd, r)
    return {"yaw": float(_wrap(yaw_)), "t": t_.tolist(), "scale": k.tolist(), "cost": float(best.fun),
            "pc": list(pc), "render_cost": c, "terms": tm}


def align_pose_scale(V, F, uvs, tex, depth, info, rgb, mask, native_log=None, n_scale_cands=2, log=print):
    """yaw 를 먼저 찾고(align_yaw: 렌더 비교), 상위 후보에서 yaw + 축별 배율 + 위치를 depth 점군 3D 정합으로 맞춘다.

    회전 대칭(symmetric) 물체는 yaw 가 무의미하므로 최선 후보 하나만 푼다.

    Returns:
        align_yaw 의 dict + scale(3,), scale_candidates, bottom_offset. yaw/t/cost/terms 는 크기 정합 결과로 바뀐다.
    """
    from scipy.spatial import cKDTree
    reg = align_yaw(V, F, uvs, tex, depth, info, rgb, mask, log=log)
    rd = reg["_renderer"]
    P = rd.obs_P_full
    if len(P) > PC_SAMPLES:
        P = P[np.random.default_rng(0).choice(len(P), PC_SAMPLES, replace=False)]
    rd.obs_P, rd.obs_tree = P, cKDTree(P)
    cands = reg["candidates"][:1 if reg["symmetric"] else n_scale_cands]
    sc = [refine_scale_depth(rd, c["yaw"], np.array(c["t"]), native_log=native_log) for c in cands]
    sc.sort(key=lambda d: d["cost"])
    b = sc[0]
    k = b["scale"]
    log(f"    scale  x{k[0]:.3f} y{k[1]:.3f} z{k[2]:.3f}  yaw {np.degrees(b['yaw']):+7.1f}°  "
        f"bottom {(b['t'][2] - reg['support_z']) * 100:+.1f} cm  cost {b['cost']:.3f} "
        f"(pc obs->mesh {b['pc'][0]:.3f}, mesh->obs {b['pc'][1]:.3f}; IoU {b['terms']['iou']:.3f})"
        + ("  others " + ", ".join(f"{np.degrees(d['yaw']):+.0f}°:{d['cost']:.3f}" for d in sc[1:])
           if len(sc) > 1 else ""))
    reg.update({"yaw": b["yaw"], "t": b["t"], "scale": b["scale"], "cost": b["cost"], "terms": b["terms"],
                "pc": b["pc"], "bottom_offset": b["t"][2] - reg["support_z"],
                "yaw_only": {"yaw": reg["yaw"], "t": reg["t"], "cost": reg["cost"], "terms": reg["terms"]},
                "scale_candidates": sc})
    return reg


def asset_pose(V, yaw, t, info):
    """바닥 중심 pose(yaw, t) -> 에셋 원점(AABB 중심) pose. (pose_world, pose_cam) 4x4."""
    R = Rotation.from_euler("z", yaw).as_matrix()
    c_local = (V.max(0) + V.min(0)) / 2
    T_world = pose_mat(R, R @ c_local + np.asarray(t))
    _, Rwc, pc = _cam(info)
    T_cam = np.linalg.inv(pose_mat(Rwc, pc)) @ T_world
    return T_world, T_cam


def render_overlay(rd, yaw, t, size=240, scale=None):
    """디버그용: (입력 크롭, 렌더(음영), 50% 겹침 + 마스크 윤곽) 3장."""
    r = rd.render(yaw, t, scale=scale)
    h, w = rd.shape
    shade = 0.45 + 0.55 * r["shade"]
    ren = np.where(r["hit"][:, None], r["rgb"] * shade[:, None], 0.0)
    occluded = r["hit"] & ~rd.obs_mask & (rd.obs_depth < r["depth"] - OCCLUDE_EPS)
    ren[occluded] *= 0.35
    obs = rd.obs_rgb
    a = (obs.reshape(h, w, 3) * 255).astype(np.uint8)
    b = (ren.reshape(h, w, 3) * 255).astype(np.uint8)
    c = np.where(r["hit"].reshape(h, w)[..., None], (a * 0.5 + b * 0.5), a).astype(np.uint8)
    cnt, _ = cv2.findContours(rd.obs_mask.reshape(h, w).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = np.ascontiguousarray(c)
    cv2.drawContours(c, cnt, -1, (0, 255, 0), 1)
    s = size / max(h, w)
    return [cv2.resize(x, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) for x in (a, b, c)]
