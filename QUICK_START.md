# 🚀 Quick Start - AMR Parser Batch Processing

Dưới đây là hướng dẫn nhanh để chạy parsing AMR trên file dữ liệu lớn.

## 1️⃣ Setup (lần đầu)

```bash
# Clone repo
git clone https://github.com/Phucgiacat/transition-amr.git
cd transition-amr

# Cài đặt environment
bash setup.sh

# ✓ Xong! Bạn sẵn sàng chạy
```

## 2️⃣ Chạy processing

### Dạng đơn giản (chạy từ đầu)
```bash
.venv/bin/python run.py \
    --input "train.en(2).jsonl" \
    --output "train-amr.txt"
```

### Dạng đầy đủ (có thể resume nếu bị ngắt)
```bash
.venv/bin/python run.py \
    --input "train.en(2).jsonl" \
    --output "train-amr.txt" \
    --process "process.json"
```

**Lợi ích**: Nếu bị ngắt (GPU crash, timeout...), chạy lại lệnh trên sẽ tiếp tục từ nơi dừng.

## 3️⃣ Kiểm tra tiến độ

```bash
.venv/bin/python progress_utils.py --show process.json
```

## 4️⃣ Các tùy chọn khác

```bash
# Sử dụng model AMR 2.0 (mặc định là AMR 3.0)
.venv/bin/python run.py --input input.jsonl --output output.txt \
    --process process.json --model "AMR2-structbart-L"

# Bắt đầu lại từ đầu (bỏ qua progress file cũ)
.venv/bin/python run.py --input input.jsonl --output output.txt \
    --process process.json --no-resume
```

---

📖 **Xem chi tiết**: Mở `PARSING_HOWTO.md`

## Định dạng file

### Input (JSONL)
```json
{"sent": "The girl travels and visits places"}
{"sent": "A dog is running in the park"}
```

### Output (Penman/AMR)
```
# ::id id0
# ::annotator bart-amr
# ::snt The girl travels and visits places
(t / travel-01
   :ARG0 (g / girl)
   :ARG4 (v / visit-01
            :ARG0 g
            :ARG1 (p / place)))
```

---

✨ **Done!** Giờ bạn có thể parse file lớn một cách an toàn với backup tiến độ.
