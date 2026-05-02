import cv2
import numpy as np
from numba import njit, prange
import sys
import os
import time
import glob

@njit
def _get_neighbors(x, y, h, w):
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


@njit
def _compute_scores(neigh, pheromone, edges, grad, alpha, beta):
    scores = np.empty(len(neigh), dtype=np.float32)
    for i in range(len(neigh)):
        nx, ny = neigh[i, 0], neigh[i, 1]
        heuristic = float(edges[nx, ny]) + float(grad[nx, ny])
        pher = float(pheromone[nx, ny]) + 1e-6
        scores[i] = (pher ** alpha) * (heuristic ** beta + 1e-6)
    return scores


@njit
def _roulette(scores):
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


@njit(parallel=True)
def run_ants(edges, grad, pheromone, edge_ys, edge_xs,
             n_ants, n_steps, alpha, beta):
    """Run all the ants in parallel.
    Return: delta_pheromone (accumulation of the trail left by each ant)"""
    h, w = edges.shape
    delta = np.zeros_like(pheromone)
    n_edge = len(edge_xs)

    for ant_i in prange(n_ants):
        if n_edge > 0:
            start_idx = int(np.random.random() * n_edge) % n_edge
            x, y = edge_ys[start_idx], edge_xs[start_idx]
        else:
            x = int(np.random.random() * h) % h
            y = int(np.random.random() * w) % w

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
# PENCERE & TRACKBAR
# ──────────────────────────────────────────────

WIN_CTRL    = "Parametreler"
WIN_ORIG    = "Orijinal"
WIN_CANNY   = "Canny (baslangic)"
WIN_ACO     = "ACO Kenar Haritasi"
WIN_OVERLAY = "Kenar Ustu (overlay)"

def nothing(_):
    pass

def setup_windows():
    for w in [WIN_CTRL, WIN_ORIG, WIN_CANNY, WIN_ACO, WIN_OVERLAY]:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 360)

    cv2.createTrackbar("Karinca sayisi",  WIN_CTRL, 60,  300, nothing)
    cv2.createTrackbar("Adim sayisi",     WIN_CTRL, 40,  150, nothing)
    cv2.createTrackbar("Alpha x10",       WIN_CTRL, 10,  50,  nothing)
    cv2.createTrackbar("Beta x10",        WIN_CTRL, 30,  100, nothing)
    cv2.createTrackbar("Buharlaşma x100", WIN_CTRL, 5,   99,  nothing)
    cv2.createTrackbar("Iterasyon",       WIN_CTRL, 20,  200, nothing)
    cv2.createTrackbar("Canny esik1",     WIN_CTRL, 60,  255, nothing)
    cv2.createTrackbar("Canny esik2",     WIN_CTRL, 140, 255, nothing)
    cv2.createTrackbar("Esikleme",        WIN_CTRL, 50,  255, nothing)
    cv2.createTrackbar("Bulaniklik",      WIN_CTRL, 5,   15,  nothing)
    cv2.createTrackbar("Overlay alfa x10",WIN_CTRL, 6,   10,  nothing)

def read_params():
    n_ants    = max(1, cv2.getTrackbarPos("Karinca sayisi",   WIN_CTRL))
    n_steps   = max(1, cv2.getTrackbarPos("Adim sayisi",      WIN_CTRL))
    alpha     = max(0.1, cv2.getTrackbarPos("Alpha x10",      WIN_CTRL) / 10.0)
    beta      = max(0.1, cv2.getTrackbarPos("Beta x10",       WIN_CTRL) / 10.0)
    evap      = max(0.01, cv2.getTrackbarPos("Buharlaşma x100", WIN_CTRL) / 100.0)
    max_iter  = max(1, cv2.getTrackbarPos("Iterasyon",        WIN_CTRL))
    canny_lo  = cv2.getTrackbarPos("Canny esik1",             WIN_CTRL)
    canny_hi  = max(canny_lo + 1, cv2.getTrackbarPos("Canny esik2", WIN_CTRL))
    thresh    = cv2.getTrackbarPos("Esikleme",                WIN_CTRL)
    blur_k    = cv2.getTrackbarPos("Bulaniklik",              WIN_CTRL)
    blur_k    = blur_k if blur_k % 2 == 1 else blur_k + 1
    blur_k    = max(3, blur_k)
    ov_alpha  = cv2.getTrackbarPos("Overlay alfa x10",        WIN_CTRL) / 10.0
    return n_ants, n_steps, alpha, beta, evap, max_iter, canny_lo, canny_hi, thresh, blur_k, ov_alpha


# ──────────────────────────────────────────────
# GÖRÜNTÜ ÖN İŞLEME
# ──────────────────────────────────────────────

MAX_DIM = 640

def preprocess(frame, blur_k, canny_lo, canny_hi):
    """Convert to grayscale, blur, compute edges and gradients."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 1)
    edges = cv2.Canny(blur, canny_lo, canny_hi).astype(np.float32) / 255.0
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    cv2.normalize(grad, grad, 0, 1, cv2.NORM_MINMAX)
    return edges, grad


def resize_if_needed(frame):
    """Resize large images to fit within MAX_DIM while keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(MAX_DIM / h, MAX_DIM / w, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


# ──────────────────────────────────────────────
# SONUÇ GÖRSELLEŞTİRME
# ──────────────────────────────────────────────

def build_result(pheromone, thresh_val):
    """Visualize the pheromone matrix and create a binary edge map by thresholding."""
    vis = cv2.GaussianBlur(pheromone, (5, 5), 0)
    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, vis_bin = cv2.threshold(vis, thresh_val, 255, cv2.THRESH_BINARY)
    return vis_bin


def build_overlay(frame, vis_bin, ov_alpha):
    """Create an overlay visualization by coloring the detected edges and blending with the original image."""
    edge_color = np.zeros_like(frame)
    edge_color[vis_bin > 0] = (0, 255, 80)
    result = cv2.addWeighted(frame, 1.0, edge_color, ov_alpha, 0)
    return result


def draw_status(img, iteration, max_iter, elapsed, n_ants, paused, finished):
    """Draw a semi-transparent status box with current parameters and instructions."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    progress = int((iteration / max(max_iter, 1)) * 100)

    if finished:
        state_txt = "TAMAMLANDI - s=kaydet  r=sifirla  q=cik"
    elif paused:
        state_txt = "DURAKLATILDI (Space=devam)"
    else:
        state_txt = f"Iterasyon: {iteration}/{max_iter}"

    lines = [
        state_txt,
        f"Ilerleme: %{progress}",
        f"Sure: {elapsed:.1f}s   Karinca: {n_ants}",
        "r=sifirla  s=kaydet  Space=duraklat",
        "Sol/Sag ok: onceki/sonraki resim",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (8, 20 + i * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 230, 180), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# OK TUŞU — PLATFORM BAĞIMSIZ ÇÖZÜM
# ──────────────────────────────────────────────

def is_left_arrow(key_raw):
    """
    BUG FIX: ok tuşu keycodeları platform'a göre değişir.
    Windows'ta waitKey(1) & 0xFF her zaman 0 döndürür (yüksek byte kaybolur).
    Çözüm: raw keycode'u (mask uygulanmamış) kontrol et.
    Linux : sol=65361, sağ=65363
    Windows: sol=2424832, sağ=2555904
    Hem Linux hem Windows'ta çalışır.
    """
    return key_raw in (65361, 2424832, 81)  # 81 = eski fallback


def is_right_arrow(key_raw):
    return key_raw in (65363, 2555904, 83)  # 83 = eski fallback


# ──────────────────────────────────────────────
# TEKİL RESİM İŞLEME DÖNGÜSÜ
# ──────────────────────────────────────────────

def process_image(frame, save_prefix="aco_result"):
    """Process a single image with ACO."""
    frame = resize_if_needed(frame)
    h, w  = frame.shape[:2]

    pheromone   = np.zeros((h, w), np.float32)
    iteration   = 0
    paused      = False
    finished    = False   # BUG FIX: tamamlanma durumunu takip et
    t_start     = time.time()
    last_params = None

    edges, grad = preprocess(frame, 5, 60, 140)
    eys = np.where(edges > 0)[0].astype(np.int32)
    exs = np.where(edges > 0)[1].astype(np.int32)
    cv2.imshow(WIN_ORIG,  frame)
    cv2.imshow(WIN_CANNY, (edges * 255).astype(np.uint8))

    while True:
        params = read_params()
        (n_ants, n_steps, alpha, beta, evap,
         max_iter, canny_lo, canny_hi, thresh_val, blur_k, ov_alpha) = params

        # Ön işleme parametreleri değişince sıfırla
        preproc_key = (canny_lo, canny_hi, blur_k)
        if last_params != preproc_key:
            edges, grad = preprocess(frame, blur_k, canny_lo, canny_hi)
            eys = np.where(edges > 0)[0].astype(np.int32)
            exs = np.where(edges > 0)[1].astype(np.int32)
            pheromone[:] = 0
            iteration    = 0
            finished     = False
            t_start      = time.time()
            last_params  = preproc_key
            cv2.imshow(WIN_ORIG,  frame)
            cv2.imshow(WIN_CANNY, (edges * 255).astype(np.uint8))

        # BUG FIX: max_iter trackbar'dan küçültülünce sıkışmayı önle
        if iteration > max_iter:
            iteration = max_iter

        # Bir iterasyon çalıştır
        if not paused and not finished:
            if iteration < max_iter:
                delta = run_ants(edges, grad, pheromone,
                                 eys, exs, n_ants, n_steps, alpha, beta)
                pheromone = (pheromone + delta) * (1.0 - evap)
                max_ph = pheromone.max()
                if max_ph > 5000:
                    pheromone /= max_ph
                iteration += 1
            else:
                # BUG FIX: tamamlandığında sonsuz döngüye girmek yerine
                # finished=True set et ve kullanıcı girdisi bekle
                finished = True
                print(f"Tamamlandı: {max_iter} iterasyon. "
                      f"s=kaydet | r=sifirla | Space=devam | q=cik")

        # Görselleştir
        vis_bin = build_result(pheromone, thresh_val)
        overlay = build_overlay(frame, vis_bin, ov_alpha)
        elapsed = time.time() - t_start

        frame_disp = frame.copy()
        draw_status(frame_disp, iteration, max_iter, elapsed, n_ants, paused, finished)

        cv2.imshow(WIN_ACO,     vis_bin)
        cv2.imshow(WIN_OVERLAY, overlay)
        cv2.imshow(WIN_ORIG,    frame_disp)

        # BUG FIX: raw keycode al (0xFF mask YOK) → ok tuşları her platformda çalışır
        key_raw = cv2.waitKey(1)
        key     = key_raw & 0xFF

        if key == ord('q'):
            return 'quit'

        elif key == ord('r'):
            pheromone[:] = 0
            iteration    = 0
            finished     = False
            t_start      = time.time()
            print("Feromon sıfırlandı.")

        elif key == ord('s'):
            ts = int(time.time())
            os.makedirs("aco_output", exist_ok=True)
            cv2.imwrite(f"aco_output/{save_prefix}_orig_{ts}.png",    frame)
            cv2.imwrite(f"aco_output/{save_prefix}_aco_{ts}.png",     vis_bin)
            cv2.imwrite(f"aco_output/{save_prefix}_overlay_{ts}.png", overlay)
            print(f"Kaydedildi: aco_output/{save_prefix}_*_{ts}.png")

        elif key == ord(' '):
            if finished:
                # Space ile devam: sıfırla ve yeniden başlat
                pheromone[:] = 0
                iteration    = 0
                finished     = False
                t_start      = time.time()
                print("Yeniden başlatıldı.")
            else:
                paused = not paused
                print("Duraklatıldı." if paused else "Devam ediyor.")

        elif is_left_arrow(key_raw):
            return 'prev'

        elif is_right_arrow(key_raw):
            return 'next'


# ──────────────────────────────────────────────
# DOSYA SEÇİM YARDIMCISI
# ──────────────────────────────────────────────

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

def collect_images(path):
    """Given a path, return a list of image file paths."""
    if os.path.isdir(path):
        files = []
        for ext in SUPPORTED_EXT:
            files += glob.glob(os.path.join(path, f"*{ext}"))
            files += glob.glob(os.path.join(path, f"*{ext.upper()}"))
        return sorted(set(files))
    elif os.path.isfile(path):
        return [path]
    else:
        return []


def pick_file_interactive():
    """Prompt the user to enter a file or directory path for processing."""
    print("Enter the image file or folder path:")
    path = input(">>> ").strip().strip('"').strip("'")
    return path


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def warmup_numba():
    """
    BUG FIX: cache=True kaldırıldı — cache'li fonksiyonlar modül dosyasından
    çağrılmadığında RuntimeError fırlatır (__pycache__ bulunamaz).
    warmup, JIT derlemesini tetikler; derleme süresi sadece ilk çalıştırmada yaşanır.
    """
    print("Numba JIT derleniyor, lütfen bekleyin...")
    d_e = np.zeros((10, 10), np.float32)
    d_g = np.zeros((10, 10), np.float32)
    d_p = np.ones((10, 10),  np.float32)
    ys  = np.zeros(1, np.int32)
    xs  = np.zeros(1, np.int32)
    run_ants(d_e, d_g, d_p, ys, xs, 2, 2, 1.0, 3.0)
    print("Numba JIT hazır.")


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = pick_file_interactive()

    image_paths = collect_images(input_path)

    if not image_paths:
        print(f"Hata: '{input_path}' konumunda desteklenen görüntü bulunamadı.")
        print(f"Desteklenen formatlar: {', '.join(SUPPORTED_EXT)}")
        sys.exit(1)

    print(f"{len(image_paths)} görüntü bulundu.")
    setup_windows()
    warmup_numba()

    idx = 0
    while 0 <= idx < len(image_paths):
        path = image_paths[idx]
        print(f"\n[{idx+1}/{len(image_paths)}] İşleniyor: {os.path.basename(path)}")

        frame = cv2.imread(path)
        if frame is None:
            print(f"  Uyarı: '{path}' okunamadı, atlanıyor.")
            idx += 1
            continue

        prefix = os.path.splitext(os.path.basename(path))[0]
        action = process_image(frame, save_prefix=prefix)

        if action == 'quit':
            break
        elif action == 'next':
            idx += 1
        elif action == 'prev':
            idx = max(0, idx - 1)
        else:
            if len(image_paths) > 1:
                print("Sonraki resim için → , çıkmak için q.")
                # BUG FIX: raw keycode ile ok tuşu kontrolü
                key_raw = cv2.waitKey(0)
                key     = key_raw & 0xFF
                if key == ord('q'):
                    break
                elif is_left_arrow(key_raw):
                    idx = max(0, idx - 1)
                else:
                    idx += 1
            else:
                idx += 1

    cv2.destroyAllWindows()
    print("Çıkıldı.")


if __name__ == "__main__":
    main()