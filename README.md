# Hệ thống phân loại chất lượng trái cây (Xử lý ảnh cổ điển, không Deep Learning)

Mọi lệnh chạy **ở thư mục gốc dự án** (nơi có `config.py`).

## Cài đặt
```
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Trạng thái
| Phần | Người | Trạng thái |
|---|---|---|
| `common/`, `config.py` | cả nhóm | ✅ |
| `m1_segmentation/` (chuẩn hoá dataset, tiền xử lý, tách nền) | TV1 | ✅ |
| `m2_features/` (màu sắc, khuyết tật, hình dạng) | TV2 | ✅ |
| `m3_classify_viz/` (phân loại, hiệu chỉnh ngưỡng, đánh giá ảnh) | TV3 | ✅ |

## Test tự động (không cần dataset thật)
```
python -m m1_segmentation.test_m1
python -m m2_features.test_m2
python -m m3_classify_viz.test_m3      # test tích hợp M1 -> M2 -> M3 trên ảnh giả lập
```

## Quy trình chạy trên dữ liệu thật (theo thứ tự)
Dữ liệu đặt ở `data/raw/Apple/{Fresh,Rotten}` và `data/raw/Orange/{Fresh,Rotten}`.

| Bước | Lệnh | Ghi chú |
|---|---|---|
| 1 | `python -m m1_segmentation.curate_dataset` | Chọn 20 ảnh calib + 30 ảnh test mỗi (quả, nhãn), letterbox 320x320 -> `data/calib`, `data/test`, `data/manifest.csv` |
| 2 | `python -m m1_segmentation.test_m1 --folder data/calib/apple/rotten --n 12` | Kiểm tra mắt thường mặt nạ tách nền (TV1) |
| 3 | `python -m m2_features.test_m2 --image data/calib/apple/rotten/apple_rotten_calib_002.png --fruit apple` (tên file đúng dạng `<quả>_<nhãn>_calib_<số>.png`, xem trong `data/calib/`) | Kiểm tra phát hiện khuyết tật trên ảnh thật (TV2) |
| 4 | `python -m m3_classify_viz.calibrate --fruit apple` | Hiệu chỉnh ngưỡng bằng ảnh calib -> `outputs/thresholds_apple.json` |
| 5 | `python -m m3_classify_viz.calibrate --fruit apple --split test` | Đánh giá trên tập test, liệt kê ảnh phân loại sai |
| 6 | Lặp bước 4-5 với `--fruit orange` | |

## Trang web minh hoạ pipeline
```bash
python web/main.py
```
Mở `http://127.0.0.1:5000`. Trang chủ có nút **Thử dataset** để chạy các ảnh trong `data/test`, chọn loại táo/cam, thanh 5 bước để chuyển M1 → M2 → M3 và ô tải ảnh riêng. Giao diện so sánh Before/After cho từng phép xử lý; `web/uploads/` là thư mục phiên làm việc cho ảnh tải lên.

Nguyên tắc: **chỉ chỉnh ngưỡng bằng tập calib**; tập test chỉ để báo cáo (tránh Overfitting, Ch.8).

## Quy ước chung
- Chỉ sửa trong thư mục module của mình; `common/` chỉ sửa khi cả nhóm đồng ý.
- Mọi ngưỡng/tham số nằm ở `config.py`; ngưỡng phân loại được `calibrate.py` ghi đè qua `outputs/thresholds_<fruit>.json`.
- Nhãn ASCII (`LOAI 1/2/3`) vì `cv2.putText` không vẽ được chữ có dấu.
- Đánh giá: Loại 1, 2 = ĐẠT; Loại 3 = KHÔNG ĐẠT; so với nhãn dataset (Fresh -> đạt, Rotten -> không đạt). Bỏ `Formalin-mixed`.

## Chỉnh khi gặp lỗi (config.py)
| Hiện tượng | Chỉnh |
|---|---|
| Tách nền sai/thiếu quả | `SEGMENT.min_dist/max_dist`, thử `SEGMENT.method = "otsu_s"` hoặc `"kmeans"`; nền bị tối dần ra góc (vignette) đã được xử lý bởi `bg_model="surface"` (mặc định) |
| Quả lành báo có vết (rìa/bóng đổ) | `DEFECT.d0_min`, `DEFECT.erode_ratio` ↑ |
| Bỏ sót vết thâm mờ | `DEFECT.d0_min` ↓ |
| Đếm quá nhiều vết nhỏ | `DEFECT.min_spot_area_ratio`, `DEFECT.open_k` ↑ |
| Điểm chói bị tính là lỗi | `DEFECT.highlight_s_max/v_min` |
| Dải màu "đạt" sai | `FRUIT_CFG[...]['ripe_hue_ranges']` |

## Hạn chế cần ghi vào báo cáo (Failure Cases)
- Vết hỏng cùng màu vỏ hoặc rất mờ sẽ bị bỏ sót; vết rất nhỏ khó phân biệt với nhiễu ảnh.
- Vùng hỏng > ~50% quả và cùng dải màu "đạt" có thể làm chọn nhầm màu lành.
- Nấm mốc trắng nhạt có thể bị loại cùng điểm chói phản quang.
- Circularity chỉ nhạy khi quả méo mạnh (tỉ lệ trục ~2:1 mới xuống ~0,75).
- Ngưỡng 1%/10% theo giáo trình không phù hợp mọi bộ dữ liệu: cần hiệu chỉnh (calibrate) và nêu rõ trong báo cáo.
