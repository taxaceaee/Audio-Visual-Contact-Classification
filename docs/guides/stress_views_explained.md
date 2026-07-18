# Stress Views: Cách Lấy, Giải Thích & Tại Sao

> Tài liệu này giải thích chi tiết cách `robot_mix` và `bandlimit` view được tạo ra từ raw audio,
> từng bước biến đổi, và lý do tại sao mỗi step được thiết kế như vậy.

## Mục lục

1. [Tổng quan — Tại sao cần stress views?](#1-tổng-quan--tại-sao-cần-stress-views)
2. [Cấu trúc chung của `apply_stress_view()`](#2-cấu-trúc-chung-của-apply_stress_view)
3. [So sánh 3 views dạng bảng](#3-so-sánh-3-views-dạng-bảng)
4. [Các hàm helper cốt lõi](#4-các-hàm-helper-cốt-lõi)
5. [Chi tiết robot_mix view](#5-chi-tiết-robot_mix-view)
6. [Chi tiết bandlimit view](#6-chi-tiết-bandlimit-view)
7. [So sánh đối chiếu từng cặp step](#7-so-sánh-đối-chiếu-từng-cặp-step)
8. [Sự khác biệt với robot_heavy (attempt 0.75)](#8-sự-khác-biệt-với-robot_heavy)
9. [Tại sao deterministic và seeded?](#9-tại-sao-deterministic-và-seeded)
10. [Ví dụ minh họa một audio qua các view](#10-ví-dụ-minh-họa)
11. [Lưu ý thực thi: stress cache](#11-lưu-ý-thực-thi-stress-cache)

---

## 1. Tổng quan — Tại sao cần stress views?

### Vấn đề

Dữ liệu huấn luyện (`hand/default`) được thu bằng micro cầm tay trong môi trường forest yên tĩnh.
Dữ liệu test (`robot/test`) được thu bằng robot — có motor noise, bandwidth hẹp hơn, distortion,
timing jitter, gain auto-ranging. Đây là **domain shift** — phân phối acoustic khác nhau hoàn toàn.

### Giải pháp

**Stress views** là các phép biến đổi deterministic (có seed) áp lên raw waveform tại train-time
để mô phỏng robot acquisition artifacts. Model được train trên `clean + robot_mix + bandlimit`
sẽ học được features invariant với những distortion này.

### Tại sao không dùng robot/test data?

Robot/test không được phép dùng trong training/validation vì sẽ gây label leakage.
Toàn bộ selection phải hoàn thành trên hand/default data. Stress views là cách duy nhất
để "nhìn thấy" robot-like distortion mà không leak.

---

## 2. Cấu trúc chung của `apply_stress_view()`

Hàm `apply_stress_view(signal, view, key, sr)` được định nghĩa trong
`train_stress_cv_select_final_test.py` (dòng 90–135). Cấu trúc:

```python
def apply_stress_view(signal, view, key, sr):
    # View 'clean': không biến đổi gì
    if view == "clean":
        return signal.astype(np.float32)

    # Seed deterministic: sha256(f"{view}|{key}") → seed 32-bit
    rng = np.random.default_rng(stable_seed(f"{view}|{key}"))
    output = signal.astype(np.float32).copy()

    # Mỗi view có pipeline riêng
    if view == "robot_mix":  ...  # 6 steps
    if view == "bandlimit": ...   # 4 steps
```

**Điểm chung:**
- Input là raw waveform 1D numpy array (float).
- Output là waveform đã biến đổi, cùng độ dài, peak-normalized về [-1, 1].
- Mỗi view dùng seed riêng = `sha256(f"{view}|{key}")` với `key = str(audio_path)`.
- Các parameter ngẫu nhiên (amplitude scale, SNR, drive...) được sample từ uniform distribution
  mỗi lần gọi, nhưng deterministic nhờ seed — cùng audio_path + cùng view → cùng kết quả.

---

## 3. So sánh 3 views dạng bảng

| Step | `clean` | `robot_mix` | `bandlimit` |
|---|---|---|---|
| FFT bandpass | None | 90–5200 Hz | 180–3600 Hz |
| Amplitude scale | None | × U(0.65, 1.25) | × U(0.75, 1.15) |
| Clipping | None | Tanh soft-clip drive U(1.15, 1.65) | Hard clip ±0.78 |
| Noise | None | Gaussian SNR U(18–28 dB) | Gaussian SNR U(24–34 dB) |
| Time roll | None | ±1200 samples (~27ms @44kHz) | None |
| Peak normalize | None | Có | Có |

**robot_mix = nặng hơn bandlimit ở mọi khía cạnh:**
- Bandpass rộng hơn (90–5200 vs 180–3600) → giữ được nhiều tần số hơn
- Distortion dạng soft-clip (giống analog saturation) thay vì hard-clip (digital)
- Noise nhiều hơn (18–28 dB vs 24–34 dB SNR)
- Có thêm time roll để mô phỏng timing jitter
- Amplitude scale biên độ rộng hơn (±35% vs ±15%)

---

## 4. Các hàm helper cốt lõi

### 4.1 `stable_seed(value: str) -> int`

```python
def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)
```

- Lấy 16 ký tự hex đầu của SHA256 → số nguyên 64-bit → mod 2^32.
- Đảm bảo cùng `value` luôn cho cùng seed (deterministic).
- `value = f"{view}|{audio_path}"` → mỗi file + mỗi view có seed riêng.

### 4.2 `fft_bandlimit(signal, sr, low_hz, high_hz) -> np.ndarray`

```python
def fft_bandlimit(signal, sr, low_hz, high_hz):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    mask = np.ones_like(freqs, dtype=bool)
    if low_hz is not None: mask &= freqs >= low_hz
    if high_hz is not None: mask &= freqs <= high_hz
    filtered = np.fft.irfft(spectrum * mask, n=len(signal))
    return filtered.astype(np.float32)
```

- Dùng `np.fft.rfft` (real FFT, chỉ tính nửa dương phổ) → nhân mask → `irfft`.
- Không dùng filter FIR/IIR thông thường vì FFT-based bandlimit là **phase-preserving**
  (zero-phase) — không gây phase shift artifacts mà filter thông thường tạo ra.
- Quan trọng: đây là **hard cut** (cửa sổ chữ nhật trong frequency domain) — có thể tạo
  ringing artifacts ở biên, nhưng chấp nhận được vì mục đích là stress test chứ không phải
  audio chất lượng cao.

### 4.3 `add_noise_at_snr(signal, rng, snr_db) -> np.ndarray`

```python
def add_noise_at_snr(signal, rng, snr_db):
    signal_rms = float(np.sqrt(np.mean(signal**2))) + 1e-8
    noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=signal.shape)
    return (signal + noise).astype(np.float32)
```

- Tính RMS của signal → suy ra RMS của noise cần để đạt SNR mong muốn: `noise_rms = signal_rms / 10^(snr_db/20)`.
- Công thức: `SNR_dB = 20 * log10(signal_rms / noise_rms)`.
- Noise là Gaussian `N(0, noise_rms)` — white noise, không có cấu trúc (khác với motor hum noise trong robot_heavy).
- `1e-8` là epsilon tránh division by zero cho silent windows.

### 4.4 `normalize_peak(signal) -> np.ndarray`

```python
def normalize_peak(signal):
    peak = float(np.max(np.abs(signal))) + 1e-8
    return (signal / peak).astype(np.float32)
```

- Chuẩn hóa về [-1, 1] dựa trên peak amplitude.
- Quan trọng: **sau tất cả biến đổi** để đảm bảo feature extraction không bị ảnh hưởng bởi gain.
  Feature sets (MFCC, total240, v.v.) đều scale theo amplitude, normalize peak giúp loại bỏ
  biến gain để model tập trung vào spectral/temporal pattern thay vì loudness.

---

## 5. Chi tiết robot_mix view

Mục tiêu: mô phỏng robot acquisition artifacts với mức độ **trung bình** — đủ nặng để
generalize lên robot thật, nhưng không quá nặng làm mất thông tin class.

```python
if view == "robot_mix":
    # Step 1: FFT bandpass 90–5200 Hz
    output = fft_bandlimit(output, sr, low_hz=90.0, high_hz=5200.0)
```

**Cách lấy:**
- Tính FFT → zero-out bins < 90 Hz và > 5200 Hz → IFFT.

**Tại sao 90–5200 Hz?**
- **90 Hz low cut:** robot microphone không capture được sub-bass.
  Trunk (thân cây) thường có năng lượng sub-bass mạnh do cộng hưởng gỗ.
  Mất sub-bass là một lý do trunk dễ bị confuse thành ambient trên robot.
- **5200 Hz high cut:** robot ADC thường có sampling rate thấp hơn hand (16kHz vs 44.1kHz),
  dẫn đến Nyquist ~8kHz. Trên thực tế bandwidth còn hẹp hơn do microphone quality.
  5200 Hz cắt bỏ ultrasonic texture — leaf (tiếng lá) mất chi tiết ma sát tần số cao.

```python
    # Step 2: Random amplitude scaling × U(0.65, 1.25)
    output = output * float(rng.uniform(0.65, 1.25))
```

**Cách lấy:**
- Sample `scale ~ Uniform(0.65, 1.25)` → nhân toàn bộ waveform với scale.

**Tại sao?**
- Robot preamp có auto-gain, không fixed gain như hand recording.
- Một leaf có thể to hoặc nhỏ tùy gain, phá vỡ tương quan `loudness ↔ class`.
- Scale ngẫu nhiên ±35% buộc model phải dựa vào spectral/temporal pattern
  thay vì amplitude tuyệt đối.

```python
    # Step 3: Gaussian noise SNR U(18–28 dB)
    output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(18.0, 28.0)))
```

**Cách lấy:**
- Sample `snr_db ~ Uniform(18, 28)` → tính noise_rms = signal_rms / 10^(snr_db/20)
  → generate Gaussian noise → cộng vào signal.

**Tại sao 18–28 dB?**
- Hand recording: SNR thường > 40 dB (forest yên tĩnh, micro tốt).
- Robot: motor noise, wheel noise, electronics hum → SNR thấp hơn nhiều.
- 18–28 dB là mức noise đủ mạnh để che lấp transient nhỏ (leaf crunch, twig snap)
  nhưng không át hẳn trunk impact.
- Ambient windows (class 0) có SNR thấp nhất → dễ bị noise lấn át → model học
  phải phân biệt giữa ambient + noise vs contact + noise.

```python
    # Step 4: Tanh soft-clipping drive U(1.15, 1.65)
    drive = float(rng.uniform(1.15, 1.65))
    output = np.tanh(drive * output) / np.tanh(drive)
```

**Cách lấy:**
- `tanh(drive * x) / tanh(drive)` là hàm soft-clip:
  - Khi |x| nhỏ (gần 0): gần như tuyến tính (gain ≈ 1).
  - Khi |x| lớn: bão hòa dần về ±1, tạo harmonics chẵn/lẻ mới.
- `drive` càng lớn → saturation càng sớm, distortion càng nhiều.

**Tại sao?**
- Robot amplifier có gain staging — tín hiệu lớn bị bão hòa (analog saturation).
- Trunk (class 2) thường có amplitude cao → bị soft-clip mạnh nhất → mất peak,
  méo waveform → dễ giống ambient hơn. Đây là mechanism chính của `trunk->ambient` error.
- Soft-clip (tanh) khác với hard-clip:

  | Hard-clip (`np.clip`) | Soft-clip (`tanh`) |
  |---|---|
  | Cắt đột ngột ở threshold | Bão hòa từ từ |
  | Tạo harmonics bậc cao | Tạo harmonics bậc thấp |
  | "Digital" sound | "Analog" sound |
  | Giữ flat-top (dễ nhận ra) | Smooth saturation (khó detect) |

  robot_mix dùng **soft-clip** vì robot amplifier saturation thường là
  analog (tube/op-amp style), không phải digital clipping.

```python
    # Step 5: Time roll ±1200 samples
    shift = int(rng.integers(-1200, 1201))
    output = np.roll(output, shift)
```

**Cách lấy:**
- `np.roll(output, shift)` dịch vòng (circular shift) toàn bộ waveform.
- `shift ~ Uniform(-1200, 1200)` integers (bao gồm cả ±1200).

**Tại sao?**
- Robot acquisition có timing jitter — trigger latency không ổn định.
- Ở 44.1kHz, ±1200 samples ≈ ±27ms. Đủ để dịch phase của transient events.
- Quan trọng cho các **pairwise models**: nếu event bị dịch đi ±27ms,
  alignment giữa các segment trong cùng specimen bị phá vỡ.
- Circular shift (không zero-pad) giữ nguyên độ dài signal, không mất context,
  nhưng tạo discontinuity tại biên — feature extractor sẽ capture artifact này.

```python
    # Step 6: Peak normalize
    return normalize_peak(output)
```

**Cách lấy:**
- `output = output / max(|output|)`.

**Tại sao?**
- Sau soft-clip và amplitude scaling, peak có thể ≠ 1.
- Normalize về [-1, 1] để feature extraction (MFCC, FFT, mel) hoạt động ổn định,
  không bị ảnh hưởng bởi gain khác nhau giữa các stress views.

---

## 6. Chi tiết bandlimit view

Mục tiêu: mô phỏng **digital transmission channel** — narrow bandwidth, hard clipping,
ít noise hơn robot_mix. Đại diện cho một robot variant khác (robot cũ, radio link, v.v.).

```python
if view == "bandlimit":
    # Step 1: FFT bandpass 180–3600 Hz
    output = fft_bandlimit(output, sr, low_hz=180.0, high_hz=3600.0)
```

**Cách lấy:** Tương tự robot_mix step 1 nhưng dải tần hẹp hơn.

**Tại sao 180–3600 Hz?**
- **180 Hz low cut:** Cao hơn robot_mix (90 Hz). Mất thêm sub-bass của trunk.
  Mô phỏng loa/ microphone rẻ tiền không tái tạo bass.
- **3600 Hz high cut:** Thấp hơn robot_mix (5200 Hz). Mất gần hết texture tần số cao
  của leaf/twig. Chỉ giữ formants và harmonic cơ bản.
- Đây là dải tần giống điện thoại (300–3400 Hz) — mô phỏng robot với codec compression.

```python
    # Step 2: Amplitude scale × U(0.75, 1.15)
    output = output * float(rng.uniform(0.75, 1.15))
```

**Tại sao ±15% thay vì ±35%?**
- Bandlimit view nhẹ hơn robot_mix ở dimension này.
- Gain variation của robot cũ/digital channel nhỏ hơn robot hiện đại.

```python
    # Step 3: Hard clip ±0.78
    output = np.clip(output, -0.78, 0.78)
```

**Cách lấy:**
- `np.clip`: mọi sample > 0.78 → 0.78, mọi sample < -0.78 → -0.78.

**Tại sao?**
- Mô phỏng **digital clipping** — ADC của robot cũ bị saturation cứng.
- Khác với soft-clip của robot_mix, hard-clip tạo flat-top waveform:
  | Before clip | After clip |
  |---|---|
  | `[0.9, -0.9, 0.95]` | `[0.78, -0.78, 0.78]` |

  Hard-clip bảo toàn phase nhưng tạo harmonics bậc cao (square wave-like),
  khác biệt rõ so với tanh saturation.

**Tại sao 0.78?**
- Là threshold hợp lý giữa gain variation (step 2) và noise floor (step 4).
- Không phải 1.0 để tạo "khoảng trống" cho noise addition không bị re-clip.

```python
    # Step 4: Gaussian noise SNR U(24–34 dB)
    output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(24.0, 34.0)))
```

**Tại sao 24–34 dB?**
- Cao hơn robot_mix (18–28 dB) → ít noise hơn, chất lượng tín hiệu tốt hơn.
- robot_mix = robot mới, nhiều motor/electronics noise hơn.
- bandlimit = robot cũ/telemetry channel — signal sạch hơn nhưng bandwidth hẹp.

```python
    # Step 5: Peak normalize (step cuối)
    return normalize_peak(output)
```

---

## 7. So sánh đối chiếu từng cặp step

### Bandpass: robot_mix rộng hơn bandlimit

```
robot_mix:  |===== 90 Hz =====|====================|===== 5200 Hz =====|
bandlimit:        |=== 180 Hz =====|============|===== 3600 Hz =====|

Sub-bass (trunk):  robot_mix còn 90–180 Hz, bandlimit mất hoàn toàn
High texture (leaf): robot_mix còn 3600–5200 Hz, bandlimit mất hoàn toàn
```

### Clipping: soft vs hard

```
robot_mix (soft-clip):  tanh → saturation từ từ, giữ smooth transition
bandlimit (hard-clip):  np.clip → cắt phẳng, tạo harmonics bậc cao

Effect trên trunk:
  - robot_mix: amplitude bị nén, waveform méo (trunk → khó phân biệt với ambient)
  - bandlimit: amplitude bị cắt phẳng, waveform vuông (trunk vẫn recognizable)
```

### Noise: robot_mix nặng hơn

```
robot_mix noise: SNR 18–28 dB → ∞ 0.13–0.40 noise/signal ratio
bandlimit noise: SNR 24–34 dB → ∞ 0.06–0.20 noise/signal ratio

18 dB SNR = noise gần bằng signal, rất khó nghe transient nhỏ
```

### Time roll: chỉ robot_mix có

```
robot_mix có roll ±27ms → phá alignment transient cho pairwise models
bandlimit không có roll → mô phỏng channel ổn định timing
```

---

## 8. Sự khác biệt với robot_heavy (attempt 0.75)

Trong `train_audio_trunk_rescue_domain_robust_select_final_test.py`,
có thêm 2 heavy views: `robot_heavy` và `pitch_speed`.

### So sánh `robot_mix` vs `robot_heavy`

| Step | robot_mix | robot_heavy |
|---|---|---|
| Reverb | None | Synthetic IR reverb (convolution với impulse response ngẫu nhiên, decay 0.03–0.08s) |
| Bandpass | 90–5200 Hz | 60–U(3800, 5200) Hz |
| Amplitude | × U(0.65, 1.25) | × U(0.5, 1.4) |
| Noise | Gaussian SNR 18–28 dB | Motor hum (45–120 Hz + 5 harmonics) + broadband SNR 10–22 dB |
| Clipping | Tanh drive 1.15–1.65 | Tanh drive 1.2–2.0 |
| Time roll | ±1200 samples | ±2000 samples |

**robot_heavy** nặng hơn robot_mix ở mọi dimension:
- Thêm **convolution reverb**: mô phỏng room acoustics / robot chassis resonance.
- **Motor hum noise** thay vì white noise: có cấu trúc (fundamental + harmonics),
  realistic hơn cho robot acquisition.
- Bandpass thấp hơn (60 Hz) để thêm sub-bass distortion, nhưng high end thay đổi
  ngẫu nhiên (3800–5200 Hz) — mô phỏng robot di chuyển qua các môi trường khác nhau.
- Drive mạnh hơn (2.0 vs 1.65) — saturation nặng hơn.

Kết quả: `robot_heavy` quá aggressive, generalization không tốt hơn `robot_mix`.
Trong attempt 0.75, nó không cải thiện được trunk→ambient errors vì
**synthetic distortion vẫn khác với robot acquisition thật**.

### `pitch_speed` view

```python
if view == "pitch_speed":
    output = speed_perturb(output, rng)   # rate U(0.85, 1.15)
    output = fft_bandlimit(output, sr, low_hz=100.0, high_hz=5000.0)
    output = output * float(rng.uniform(0.7, 1.3))
    output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(18.0, 30.0)))
    return normalize_peak(output)
```

- `speed_perturb`: linear interpolation stretch/compress ±15%.
  Thay đổi cả pitch và duration — mô phỏng robot tốc độ khác nhau.
- Đây là stress view khác dimension: **temporal distortion** thay vì spectral.

---

## 9. Tại sao deterministic và seeded?

### Cơ chế

```python
def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)

rng = np.random.default_rng(stable_seed(f"{view}|{key}"))
```

Với `key = str(audio_path)`:
- Cùng audio_path + cùng view → luôn cùng seed → cùng kết quả.
- Khác audio_path → seed khác (dù cùng view).
- Khác view (robot_mix vs bandlimit) → seed khác (dù cùng audio_path).

### Tại sao deterministic?

1. **Cacheability:** Cùng file luôn cho cùng stress features → có thể cache `X.npy`, `y.npy`
   và reuse (see stress cache). Nếu random mỗi lần, cache vô dụng.
2. **Reproducibility:** Training pipeline có thể chạy lại cho cùng kết quả — quan trọng
   cho paper/science.
3. **Fold consistency:** Trong cross-validation, mỗi fold dùng cùng stress views —
   không bị data leakage giữa các fold do random khác nhau.

### Tại sao seed riêng cho mỗi audio_path?

Nếu dùng global seed cho tất cả files:
- Tất cả files bị cùng amplitude scale (vd tất cả × 0.7).
- Tất cả files bị cùng shift (±500 samples).
- Tất cả files bị cùng noise pattern.

Dùng seed riêng mỗi file:
- Mỗi file bị biến đổi khác nhau — tạo training data diverse hơn.
- Model không thể "học" cái pattern noise cố định → generalize tốt hơn.

---

## 10. Ví dụ minh họa

### trunk window (class 2) qua các views

Giả sử trunk window có spectral năng lượng tập trung ở sub-bass (50–200 Hz)
với amplitude cao.

| View | Spectral change | Amplitude change | Kết quả |
|---|---|---|---|
| **clean** | Full spectrum (20 Hz–22 kHz) | 0 dB gain | Trunk rõ: sub-bass + high amplitude |
| **robot_mix** | Mất <90 Hz, >5200 Hz | Soft-clip giảm peak ~20% | Trunk mất sub-bass, amplitude thấp hơn → dễ confuse ambient |
| **bandlimit** | Mất <180 Hz, >3600 Hz | Hard-clip ±0.78 cắt phẳng | Trunk mất sub-bass, waveform vuông → vẫn nhận ra |
| **robot_heavy** | Mất <60 Hz, >U(3800-5200) Hz, thêm reverb | Soft-clip mạnh drive 2.0 | Trunk bị reverb làm mờ transient, noise motor che sub-bass |

### leaf window (class 1) qua các views

Leaf có spectral texture tần số cao (ma sát lá) với amplitude thấp.

| View | Spectral change | Noise effect | Kết quả |
|---|---|---|---|
| **clean** | Texture tần số cao rõ | Không noise | Leaf rõ: high-freq crunch |
| **robot_mix** | Mất >5200 Hz → mất texture | SNR 18 dB → noise ~40% signal | Leaf texture bị noise che → dễ confuse ambient |
| **bandlimit** | Mất >3600 Hz → mất gần hết texture | SNR 24 dB → cleaner | Leaf gần như mất hết đặc trưng → rất khó |
| **robot_heavy** | Mất >U(3800-5200) Hz | Motor hum + noise SNR 10 dB | Leaf gần như không detect được |

---

## 11. Lưu ý thực thi: stress cache

```python
def build_or_load_stress_cache(frame, view, stress_feature_dir, force_rebuild):
```

Sau khi stress view được apply, features được extract và cache vào disk:

```
stress_features/
  robot_mix/
    X.npy           # feature matrix (n_samples × n_features)
    y.npy           # labels
    paths.npy       # audio paths
    metadata.json   # feature_set, feature_dim, manifest_signature, extraction_time
  bandlimit/
    ... (tương tự)
```

**Metadata validation:**
```python
valid_metadata = (
    metadata["feature_set"] == FEATURE_SET
    and metadata["feature_dim"] == EXPECTED_DIM
    and metadata["n_samples"] == len(frame)
    and metadata["manifest_signature"] == signature
    and metadata["stress_view"] == view
)
```

Cache chỉ được dùng lại nếu **tất cả** match:
- Cùng feature set (total240, mfcc40, v.v.)
- Cùng số chiều features
- Cùng số samples
- Cùng manifest (thay đổi file list → rebuild)
- Cùng stress view name

Điều này ngăn silent bugs khi config thay đổi mà cache cũ vẫn được dùng.

---

## Tóm tắt

| | robot_mix | bandlimit |
|---|---|---|
| **Mục đích** | Mô phỏng robot acquisition distortion (chất lượng trung bình) | Mô phỏng digital narrow channel (robot cũ / telemetry) |
| **Bandpass** | 90–5200 Hz (giữ được sub-bass cơ bản và high texture) | 180–3600 Hz (hẹp, mất sub-bass và high texture) |
| **Clipping** | Soft-clip (tanh) — analog saturation | Hard-clip (±0.78) — digital clipping |
| **Noise** | Gaussian SNR 18–28 dB (nặng) | Gaussian SNR 24–34 dB (nhẹ hơn) |
| **Timing jitter** | ±1200 samples (~27ms roll) | None |
| **Độ nặng** | Trung bình | Nhẹ hơn robot_mix (trừ bandpass hẹp hơn) |
| **Số steps** | 6 | 4 |
| **Dùng trong** | Training & TTA của HGB anchor, pairwise models, high-SR models | Training & TTA của high-SR models, stress-CV selection |

Cả hai view cùng tạo ra **domain-invariant features**: model học từ `clean + robot_mix + bandlimit`
sẽ không bị overfit vào bất kỳ acquisition condition nào, và hy vọng generalize lên robot thật.
