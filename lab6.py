import numpy as np
import zlib
import matplotlib.pyplot as plt
from PIL import Image

# Базова таблиця квантування JPEG (Luminance Q50)
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

# --- ДОПОМІЖНІ ФУНКЦІЇ ---

def blockify(img):
    h, w = img.shape
    return img.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)

def unblockify(blocks, h, w):
    return blocks.reshape(h // 8, w // 8, 8, 8).swapaxes(1, 2).reshape(h, w)

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

# --- ОСНОВНІ ФУНКЦІЇ ЛАБОРАТОРНОЇ ---

def dct_matrix(N=8):
    D = np.zeros((N, N))
    for k in range(N):
        a = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            D[k, n] = a * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return D

D_GLOBAL = dct_matrix(8)

def dct2(block):
    return D_GLOBAL @ block @ D_GLOBAL.T

def idct2(coeffs):
    return D_GLOBAL.T @ coeffs @ D_GLOBAL

def qtable(q):
    q = max(1, min(99, q))
    S = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((S * Q50 + 50) / 100), 1, 255)

def encode(img, q):
    Q = qtable(q)
    blocks = blockify(img.astype(np.float64) - 128)
    coeffs = np.array([np.round(dct2(b) / Q) for b in blocks])
    return coeffs.astype(np.int16), Q

def decode(coeffs, Q, h, w):
    blocks = np.array([idct2(c * Q) for c in coeffs])
    img = unblockify(blocks, h, w) + 128
    return np.clip(img, 0, 255).round().astype(np.uint8)

# --- ТЕСТИ ТА ГЕНЕРАЦІЯ ЗВІТУ ---

def run_tests():
    print("=== ЗАПУСК ТЕСТІВ ===")
    test_block = np.random.randint(0, 256, (8, 8)).astype(np.float64) - 128
    reconstructed = idct2(dct2(test_block))
    err = np.max(np.abs(test_block - reconstructed))
    assert err < 1e-12, f"Помилка DCT/IDCT: {err}"
    print("[ОК] Тест 1-2: Пряме та зворотне DCT працюють коректно")
    
    assert np.array_equal(qtable(50), Q50), "qtable(50) не збігається з Q50!"
    print("[ОК] Тест 3: Таблиця qtable(50) відповідає Q50")
    
    dummy_img = np.full((512, 512), 128, dtype=np.uint8)
    dummy_img[200:300, 200:300] = 200
    coeffs, Q = encode(dummy_img, 50)
    rec_img = decode(coeffs, Q, 512, 512)
    assert calculate_psnr(dummy_img, rec_img) > 30
    print("[ОК] Тест 4-5: Кодер та Декодер працюють успішно")
    print("=== УСІ ТЕСТИ ПРОЙДЕНО ===\n")

def generate_report_data():
    x, y = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512))
    img = (255 * (0.5 * np.sin(10 * x) + 0.5 * np.cos(10 * y))).astype(np.uint8)

    h, w = img.shape
    raw_size = h * w
    qualities = [95, 90, 75, 50, 30, 15, 5]
    results = []
    
    print(f"{'q':<5}{'Байт':<10}{'Стиснення':<12}{'Нулів (%)':<12}{'PSNR (dB)':<12}{'SSIM':<10}")
    print("-" * 62)
    
    for q in qualities:
        coeffs, Q = encode(img, q)
        rec_img = decode(coeffs, Q, h, w)
        
        compressed_bytes = len(zlib.compress(coeffs.tobytes()))
        compression_ratio = raw_size / compressed_bytes
        zeros_pct = (np.sum(coeffs == 0) / coeffs.size) * 100
        
        psnr_v = calculate_psnr(img, rec_img)
        ssim_v = calculate_ssim(img, rec_img)
        
        results.append((q, compressed_bytes, compression_ratio, zeros_pct, psnr_v, ssim_v))
        print(f"{q:<5}{compressed_bytes:<10}{compression_ratio:<12.1f}x{zeros_pct:<12.1f}%{psnr_v:<12.1f}{ssim_v:<10.3f}")
        
        Image.fromarray(rec_img).save(f"q{q:02d}.png")
        
    qs = [r[0] for r in results]
    psnrs = [r[4] for r in results]
    sizes = [r[1] / 1024 for r in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'blue'
    ax1.set_xlabel('Якість (q)')
    ax1.set_ylabel('PSNR (dB)', color=color)
    ax1.plot(qs, psnrs, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Розмір (КБ)', color=color)
    ax2.plot(qs, sizes, 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Крива якості JPEG-lite')
    fig.tight_layout()
    plt.savefig('curve.png', dpi=300)
    print("\nГрафік успішно збережено у 'curve.png'.")

if __name__ == "__main__":
    run_tests()
    generate_report_data()