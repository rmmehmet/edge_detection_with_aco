import cv2
import numpy as np
from numba import njit, prange
import time
import os

@njit(cache=True)
def _compute_scores(neigh, pheromone, edges, grad, alpha, beta):
    """Calculate neighboring pixel scores
     Score = (pheromone^alpha) * (heuristic^beta)"""
    scores = np.empty(len(neigh), dtype=np.float32)
    for i in range(len(neigh)):
        nx, ny = neigh[i, 0], neigh[i, 1]
        heuristic = float(edges[nx, ny]) + float(grad[nx, ny])
        pher = float(pheromone[nx, ny]) + 1e-6  # sıfır bölme önle
        scores[i] = (pher ** alpha) * (heuristic ** beta + 1e-6)
    return scores

@njit(cache=True)
def _get_neighbors(x, y, h, w):
    """Return an 8-neighborhood pixel list
    Returns 8 neighboring coordinates around (x,y)"""
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    buf = np.empty((8, 2), dtype=np.int32)
    count = 0
    for k in range(8):
        nx = x + moves[k][0]
        ny = y + moves[k][1]
        if 0 <= nx < h and 0 <= ny < w:
            buf[count, 0] = nx
            buf[count, 1] = ny
            count += 1
    return buf[:count]

@njit(cache=True)
def _roulette(scores):
    """Probabilistic choice (roulette wheel)
     Return: index of chosen neighbor"""
    total = scores.sum()
    if total == 0.0:
        return 0
    r = np.random.random() * total
    cumsum = 0.0
    for i in range(len(scores)):
        cumsum += scores[i]
        if cumsum >= r:
            return i
    return len(scores) - 1

@njit(cache=True, parallel=True)
def run_ants(edges, grad, pheromone, edge_ys, edge_xs,
             n_ants, n_steps, alpha, beta):
    """
    Run all the ants in parallel.
    Return: delta_pheromone (accumulation of the trail left by each ant)
    """
    h, w = edges.shape
    delta = np.zeros_like(pheromone)
    n_edge = len(edge_xs)

    for ant_i in prange(n_ants):
        # Başlangıç noktası: kenar pikseli seç
        if n_edge > 0:
            start_idx = int(np.random.random() * n_edge) % n_edge
            x, y = edge_ys[start_idx], edge_xs[start_idx]
        else:
            x = int(np.random.random() * h) % h
            y = int(np.random.random() * w) % w

        # Yürüyüş
        for step in range(n_steps):
            neigh = _get_neighbors(x, y, h, w)
            if len(neigh) == 0:
                break
            scores = _compute_scores(neigh, pheromone, edges, grad, alpha, beta)
            chosen = _roulette(scores)
            x, y = neigh[chosen, 0], neigh[chosen, 1]
            delta[x, y] += 1.0

    return delta

# ──────────────────────────────────────────────
# TRACKBAR WINDOW SETUP
# ──────────────────────────────────────────────

WIN_CTRL  = "Parameters"
WIN_ORIG  = "Original"
WIN_CANNY = "Canny (initial)"
WIN_ACO   = "ACO Edge Map"

def nothing(_):
    pass

def setup_windows():
    cv2.namedWindow(WIN_CTRL,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_ORIG,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CANNY, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_ACO,   cv2.WINDOW_NORMAL)

    cv2.resizeWindow(WIN_CTRL, 400, 320)

    # Trackbar'lar
    cv2.createTrackbar("Number of Ants",  WIN_CTRL, 40,  200, nothing)
    cv2.createTrackbar("Number of Steps",     WIN_CTRL, 30,  100, nothing)
    cv2.createTrackbar("Alpha x10",       WIN_CTRL, 10,  50,  nothing)  # 1.0
    cv2.createTrackbar("Beta x10",        WIN_CTRL, 30,  100, nothing)  # 3.0
    cv2.createTrackbar("Evaporation x100", WIN_CTRL, 10,  99,  nothing)  # 0.10
    cv2.createTrackbar("Canny threshold1",     WIN_CTRL, 60,  255, nothing)
    cv2.createTrackbar("Canny threshold2",     WIN_CTRL, 140, 255, nothing)
    cv2.createTrackbar("Threshold",        WIN_CTRL, 50,  255, nothing)
    cv2.createTrackbar("Blur",      WIN_CTRL, 5,   15,  nothing)  # GaussianBlur kernel size

def read_params():
    n_ants     = max(1,  cv2.getTrackbarPos("Number of Ants",  WIN_CTRL))
    n_steps    = max(1,  cv2.getTrackbarPos("Number of Steps",     WIN_CTRL))
    alpha      = max(0.1, cv2.getTrackbarPos("Alpha x10",       WIN_CTRL) / 10.0)
    beta       = max(0.1, cv2.getTrackbarPos("Beta x10",        WIN_CTRL) / 10.0)
    evap       = max(0.01, cv2.getTrackbarPos("Evaporation x100", WIN_CTRL) / 100.0)
    canny_lo   = cv2.getTrackbarPos("Canny esik1",     WIN_CTRL)
    canny_hi   = max(canny_lo + 1, cv2.getTrackbarPos("Canny esik2", WIN_CTRL))
    thresh_val = cv2.getTrackbarPos("Threshold",        WIN_CTRL)
    blur_k     = cv2.getTrackbarPos("Bulaniklik",      WIN_CTRL)
    blur_k     = blur_k if blur_k % 2 == 1 else blur_k + 1  
    blur_k     = max(3, blur_k)
    return n_ants, n_steps, alpha, beta, evap, canny_lo, canny_hi, thresh_val, blur_k

# ──────────────────────────────────────────────
# OVERLAY: FPS + PARAMETER DISPLAY
# ──────────────────────────────────────────────
def draw_overlay(img, fps, n_ants, n_steps, alpha, beta, evap):
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (310, 115), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    lines = [
        f"FPS: {fps:.1f}",
        f"Karinca: {n_ants}  Adim: {n_steps}",
        f"Alpha: {alpha:.1f}  Beta: {beta:.1f}",
        f"Buharlaşma: {evap:.2f}",
        "q=cik  r=sifirla  s=kaydet",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (8, 20 + i*19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,230,200), 1, cv2.LINE_AA)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Webcam açılamadı.")
        return

    setup_windows()

    pheromone = None
    prev_gray = None
    frame_count = 0
    t_prev = time.time()
    fps = 0.0
    save_dir = "screenshots"

    print("ACO Webcam started. q=exit | r=reset | s=save")

    dummy_e = np.zeros((10,10), np.float32)
    dummy_g = np.zeros((10,10), np.float32)
    dummy_p = np.ones((10,10),  np.float32)
    ys, xs  = np.where(dummy_e > 0)
    run_ants(dummy_e, dummy_g, dummy_p, ys.astype(np.int32), xs.astype(np.int32),
             2, 2, 1.0, 3.0)
    print("Numba JIT derlendi.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (n_ants, n_steps, alpha, beta, evap,
         canny_lo, canny_hi, thresh_val, blur_k) = read_params()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 1)

        # Kenar haritası
        edges = cv2.Canny(blur, canny_lo, canny_hi).astype(np.float32) / 255.0

        # Gradyan büyüklüğü
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        cv2.normalize(grad, grad, 0, 1, cv2.NORM_MINMAX)

        h, w = gray.shape

        # Feromon başlat / boyut uyumsuzluğunda sıfırla
        if pheromone is None or pheromone.shape != (h, w):
            pheromone = np.zeros((h, w), np.float32)

        # Adaptif sıfırlama: ani sahne değişimi
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            if diff > 25.0:
                pheromone *= 0.2

        prev_gray = gray.copy()

        # Kenar piksel indeksleri
        eys, exs = np.where(edges > 0)
        eys = eys.astype(np.int32)
        exs = exs.astype(np.int32)

        # ACO çalıştır (Numba paralel)
        delta = run_ants(edges, grad, pheromone,
                         eys, exs, n_ants, n_steps, alpha, beta)

        # Güncelle + buharlaştır
        pheromone = (pheromone + delta) * (1.0 - evap)

        # Taşma önleme
        max_ph = pheromone.max()
        if max_ph > 5000:
            pheromone /= max_ph

        # Görselleştir
        vis = cv2.GaussianBlur(pheromone, (5,5), 0)
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, vis_bin = cv2.threshold(vis, thresh_val, 255, cv2.THRESH_BINARY)

        # FPS hesapla
        frame_count += 1
        if frame_count % 10 == 0:
            t_now = time.time()
            fps = 10.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

        # Overlay
        frame_disp = frame.copy()
        draw_overlay(frame_disp, fps, n_ants, n_steps, alpha, beta, evap)

        # Göster
        cv2.imshow(WIN_ORIG,  frame_disp)
        cv2.imshow(WIN_CANNY, (edges * 255).astype(np.uint8))
        cv2.imshow(WIN_ACO,   vis_bin)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pheromone[:] = 0
            print("Pheromone map reset.")
        elif key == ord('s'):
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time())
            cv2.imwrite(f"{save_dir}/orig_{ts}.png",  frame)
            cv2.imwrite(f"{save_dir}/canny_{ts}.png", (edges*255).astype(np.uint8))
            cv2.imwrite(f"{save_dir}/aco_{ts}.png",   vis_bin)
            print(f"Kaydedildi: {save_dir}/*_{ts}.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()