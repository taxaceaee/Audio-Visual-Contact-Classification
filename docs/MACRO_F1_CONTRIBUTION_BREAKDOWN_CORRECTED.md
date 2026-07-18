# Macro-F1 Contribution Breakdown — Corrected Report

**Date:** 2026-07-18  
**Goal:** Xác định đóng góp thật sự làm tăng Macro F1 (4-class, robot/test), sau khi phát hiện breakdown theo diagram-node trước đó bị sai phương pháp.  
**Selection protocol:** pure `val_macro_f1_4class`, lock-before-test, group-safe split, seed 42.

---

## 1. Root cause: vì sao breakdown cũ “phẳng” / vô nghĩa

### Triệu chứng
Các node diagram (Anchor Blend, Report Gate, Final Source Blend, Segment Aggregation, Segment Consensus, Contact mass) khi chạy **feature-level fuse+select** cho **cùng một con số** robot Macro F1:

| Experiment slug | `fused_dim` | views | Full Macro F1 | Selected-k1200 Macro F1 |
|---|---:|---|---:|---:|
| `audio_anchor_blend_fuse_select` | 1440 | SIX_VIEWS total240 | **0.5583** | **0.5633** |
| `audio_report_gate_fuse_select` | 1440 | same | **0.5583** | **0.5633** |
| `audio_final_source_blend_fuse_select` | 1440 | same | **0.5583** | **0.5633** |
| `audio_segment_aggregation_fuse_select` | 1440 | same | **0.5583** | **0.5633** |
| `audio_segment_consensus_fuse_select` | 1440 | same | **0.5583** | **0.5633** |
| `audio_contact_mass_fuse_select` | 1440 | same | **0.5583** | **0.5633** |

Snapshot: `outputs/audio_feature_benchmarks/macro_f1_nested_contribution_breakdown/reports/invalid_diagram_node_family_snapshot.json`

### Nguyên nhân chính (PRIMARY)
**Cùng một ma trận feature.** Mọi slug trên dùng `SIX_VIEWS` total240 → `fused_dim=1440`, cùng seed 42, cùng protocol train/select/refit. Đổi **tên node** không đổi input ⇒ model + pred + F1 trùng tuyệt đối. Đây không phải “không có contribution” — mà là **thiết kế ablation sai**: không có biến độc lập.

### Nguyên nhân phụ (SECONDARY)
Các stage diagram **Anchor / Report Gate / Segment pool / Consensus / Lift** trong paper pipeline là **xử lý trên probability** (postprocess), **không** phải feature fusion mới. Gắn chúng vào `audio_*_fuse_select` với feature 1440-D chỉ đo lại baseline six-view HistGB, không đo postprocess.

### Hệ quả
Không được dùng bảng diagram-node feature-level kia để “breakdown contribution”. Cần **ladder có input khác nhau** (nested features) và/hoặc **ladder proba** (đúng stage diagram).

---

## 2. Phương pháp đã sửa

Hai ladder **hợp lệ**, chạy lại bằng `run_macro_f1_nested_contribution_breakdown.py`:

### A. Nested feature scopes (input thật sự đổi)
| Stage | Input | Dim |
|---|---|---:|
| `highsr_3view_720` | high-SR clean + robot_mix + bandlimit | 720 |
| `sr16_3view_720` | 16 kHz clean + robot_mix + bandlimit | 720 |
| `sixview_1440` | fuse cả hai block 3-view | 1440 |
| `multimodal_1952` | sixview + frozen ResNet18-224 (512-D) | 1952 |

Mỗi stage: group-safe train/val → pure val Macro-F1 model select → refit full hand → **một** lần eval robot.  
**Lưu ý:** `highsr → sr16` là so sánh **hai block song song** (không additive); nested thật sự bắt đầu từ `sixview` (gộp hai block) rồi `multimodal` (thêm ảnh).

### B. Paper-safe proba component ladder
Reuse CSV đã lock từ paper breakdown (cùng test set n=2219):

`outputs/audio_paper_breakdown_20260707/contribution_ablation/test_component_ladder.csv`

Đây mới là attribution cho các node diagram (segment pool, consensus, report gate, …).

### Artefacts
```
outputs/audio_feature_benchmarks/macro_f1_nested_contribution_breakdown/reports/
  root_cause_diagnosis.json
  nested_feature_ladder.csv | .json
  macro_f1_deltas.csv | .json
  proba_component_ladder_test.csv
  proba_component_ladder_oof.csv
  invalid_diagram_node_family_snapshot.json
  run_summary.json
  stage_*.json
```

Tests: `tests/test_macro_f1_nested_contribution.py` — 3/3 passed.

---

## 3. Kết quả ladder A — Nested features (robot/test, n=2219)

| Stage | Dim | Model (val select) | Val Macro F1 | **Test Macro F1** | Δ prev | Δ from highsr |
|---|---:|---|---:|---:|---:|---:|
| highsr_3view_720 | 720 | HistGB | 0.8001 | **0.5434** | — | 0 |
| sr16_3view_720 | 720 | HistGB | 0.8171 | **0.5636** | **+0.0202** | +0.0202 |
| sixview_1440 | 1440 | HistGB | 0.8203 | **0.5583** | **−0.0053** | +0.0150 |
| multimodal_1952 | 1952 | ExtraTrees | 0.9976 | **0.7053** | **+0.1470** | **+0.1619** |

### Đọc contribution (feature-level)
1. **16 kHz 3-view alone** tốt hơn high-SR 3-view alone trên robot (~+2.0 Macro F1). High-SR TTA views **không** tự thắng 16 kHz block trong setup fuse+HistGB này.
2. **Fuse six-view (1440)** so với best 3-view (sr16): **không tăng** Macro F1 (−0.5 pp). Val hơi cao hơn nhưng robot không cải — dấu hiệu redundancy / domain gap, không phải “gộp view = luôn tốt hơn”.
3. **Multimodal (+ResNet18 512-D)** là **bước nhảy feature-level lớn nhất**: sixview 0.558 → **0.705** (**+0.147**). Gần như toàn bộ gain feature-level từ baseline highsr → top nằm ở **image branch**.
4. Số sixview 0.5583 **trùng** invalid diagram-node family — xác nhận family kia chỉ là sixview HistGB được dán nhãn lại.

**Kết luận ladder A:** đóng góp Macro F1 thật ở tầng feature chủ yếu đến từ **multimodal image fusion**, không từ việc đổi tên node diagram trên cùng 1440-D audio.

---

## 4. Kết quả ladder B — Proba postprocess (paper stack, robot/test)

Đây mới là breakdown **đúng ngữ nghĩa diagram** (thao tác trên probability của pipeline audio paper, không retrain fuse feature).

| # | Stage | Macro F1 | Δ prev | Ghi chú |
|---|---|---:|---:|---|
| 01 | highsr_raw (window argmax) | **0.5731** | — | baseline source |
| 02 | pairwise_raw | 0.5377 | −0.0354 | pairwise alone yếu hơn |
| 03 | highsr80 + pairwise20 raw blend | **0.5764** | **+0.0387** | blend phục hồi / nhỉnh baseline |
| 04 | + segment-level log-prob pooling | **0.6393** | **+0.0629** | **gain lớn #1** |
| 05 | + specimen contact consensus (no lift) | **0.6907** | **+0.0514** | **gain lớn #2** |
| 06 | + ambient→contact lift (anchor) | 0.6938 | +0.0032 | nhỏ |
| 07 | + report_gate one-hot blend (0.05) | 0.7000 | +0.0061 | nhỏ–vừa |
| 08 | + post-blend segment lift (final) | **0.7027** | +0.0027 | nhỏ |

**Tổng stack proba:** 0.5731 → **0.7027** ≈ **+0.1295** Macro F1.

### Đóng góp proba theo nhóm
| Nhóm | Stages | Σ Δ Macro F1 (xấp xỉ) |
|---|---|---:|
| Source blend (highsr↔pairwise) | 02–03 | ~+0.003 net vs 01 (qua 02 dip) |
| **Segment pooling** | 04 | **+0.063** |
| **Specimen consensus** | 05 | **+0.051** |
| Anchor lift + report gate + final lift | 06–08 | **+0.012** |

**Kết luận ladder B:** trong diagram audio paper, Macro F1 tăng thật chủ yếu từ **segment pooling** và **specimen-level consensus**; Report Gate / Anchor lift / final lift chỉ tinh chỉnh nhỏ (~1.2 pp gộp). Gắn các node này vào feature fuse 1440-D **không** đo được các Δ này.

---

## 5. Tổng hợp: Macro F1 thật sự đến từ đâu?

```
[INVALID] Diagram-node feature-level (Anchor/RG/FSB/SegAgg/SegCons/Contact)
          → same 1440-D × seed 42 → F1 flat 0.5583  ❌ không attribution

[VALID-A] Feature nesting
          highsr 0.543  ──(+0.02)──► sr16 0.564
                 └──── fuse sixview 0.558 (≈ no gain vs best 3-view)
                                    └──(+0.147)──► multimodal 0.705  ★ feature-level winner

[VALID-B] Proba diagram stack (paper)
          highsr raw 0.573
            → blend ~0.576
            → segment pool 0.639   ★
            → consensus 0.691      ★
            → anchor/RG/lift 0.703 (small polish)
```

### Ranking đóng góp (thực dụng)
1. **Image / multimodal fusion** (+~0.15 Macro F1 trên sixview audio) — ladder A.  
2. **Segment pooling** (+~0.063) — ladder B.  
3. **Specimen consensus** (+~0.051) — ladder B.  
4. **16 kHz vs high-SR 3-view alone** (~+0.02 cho 16 kHz) — ladder A.  
5. **Report gate / anchor lift / final lift** (~+0.01 gộp) — ladder B.  
6. **Six-view fuse vs best 3-view alone** (~0 hoặc âm trên robot, pure HistGB) — ladder A.  
7. **Contact mass / “node rename” feature runs** — **0 contribution measurable** vì cùng input.

Hai path top-line khác nhau về semantic:
- Multimodal ExtraTrees ~**0.705** (feature fuse + image, **không** qua full proba stack diagram).  
- Paper proba final ~**0.703** (audio postprocess stack, **không** bắt buộc multimodal feature fuse trong CSV này).

Không cộng dồn mù quáng hai ladder (khác model path / postprocess). Dùng từng ladder đúng câu hỏi:
- “Thêm feature gì giúp F1?” → A  
- “Stage diagram proba nào giúp F1?” → B  

---

## 6. Cross-check structural

| Check | Result |
|---|---|
| Invalid family: cùng `fused_dim=1440` + cùng Macro F1 | PASS (6/6 = 0.5583 / 0.5633) |
| Nested stages dims | PASS `[720, 720, 1440, 1952]` |
| Nested stages **không** cùng test F1 | PASS (0.543 / 0.564 / 0.558 / 0.705) |
| Pure val Macro select, test not loaded at selection | PASS |
| Proba ladder loaded (test + oof) | PASS |
| Unit/entry tests | PASS (3/3) |

Logs: `/tmp/grok-goal-1cee5476899b/implementer/{breakdown_root_cause,nested_full_run,crosscheck_nested,nested_tests_rerun}.log`

---

## 7. Cách tái chạy

```bash
# tests
/tmp/grok-goal-1cee5476899b/implementer/venv/bin/python -m pytest \
  tests/test_macro_f1_nested_contribution.py -v

# full nested + attach proba CSVs
/tmp/grok-goal-1cee5476899b/implementer/venv/bin/python \
  run_macro_f1_nested_contribution_breakdown.py
```

---

## 8. Takeaways ngắn cho user

1. Breakdown diagram-node trước **có vấn đề** vì **cùng feature matrix 1440-D**, không vì selection bug.  
2. Đã sửa bằng **nested feature ladder** + **proba component ladder**, re-run full, có deltas.  
3. **Feature-level:** image (multimodal) ≈ toàn bộ jump lớn; six-view fuse không beat sr16 alone.  
4. **Proba diagram:** segment pooling + specimen consensus ≈ 90% gain stack; report gate / lift chỉ polish.  
5. Muốn “đóng góp từng node diagram” từ nay **chỉ** dùng proba ladder (hoặc ablation postprocess), **không** fuse-select lại cùng SIX_VIEWS.
