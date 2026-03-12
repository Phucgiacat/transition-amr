## 📚 Hướng dẫn sử dụng script parsing AMR

### 📋 Tổng quan

Thư mục này chứa các script Python để chạy IBM Transition AMR Parser trên tập dữ liệu lớn với hỗ trợ:
- **Resume từ checkpoint**: Nếu bị ngắt (GPU crash, timeout, etc.), có thể tiếp tục từ nơi dừng
- **Theo dõi tiến độ**: File `process.json` lưu trạng thái xử lý hiện tại
- **Xử lý lỗi**: Ghi lại các câu không thể parse và lý do tại sao

### 🛠️ Cài đặt environment

```bash
# Chạy setup script (một lần)
bash setup.sh

# Hoặc cài đặt thủ công
uv venv .venv --python 3.8
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate  # Windows

# Sau đó cài dependencies
uv pip install pip setuptools wheel matplotlib ipdb progressbar2 penman
uv pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1
uv pip install 'numpy>=1.19,<1.20'
uv pip install fairseq==0.10.2
uv pip install -e .
uv pip install --no-index torch-scatter \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

### 🚀 Sử dụng cơ bản

#### **Option 1: Chạy từ đầu (không lưu tiến độ)**

```bash
.venv/bin/python run.py \
    --input "train.en(2).jsonl" \
    --output "train-amr.txt"
```

- Input: File JSONL với các câu cần parse
- Output: File text với AMR được định dạng Penman
- Không có file tiến độ → Chạy từ câu đầu tiên

#### **Option 2: Chạy với progress tracking (có thể resume)**

```bash
.venv/bin/python run.py \
    --input "train.en(2).jsonl" \
    --output "train-amr.txt" \
    --process "process.json"
```

- Lần chạy đầu: Tạo `process.json` với tiến độ
- Nếu bị ngắt: Chạy lại lệnh trên sẽ **tiếp tục từ nơi dừng**
- Không cần `--no-resume` nếu muốn tiếp tục

#### **Option 3: Bắt đầu lại từ đầu**

```bash
.venv/bin/python run.py \
    --input "train.en(2).jsonl" \
    --output "train-amr.txt" \
    --process "process.json" \
    --no-resume
```

- Bỏ qua `process.json` hiện tại
- Sẽ ghi đè file output từ đầu

#### **Option 4: Sử dụng model khác**

```bash
.venv/bin/python run.py \
    --input "train.en(2).jsonl" \
    --output "train-amr.txt" \
    --process "process.json" \
    --model "AMR2-structbart-L"
```

- Mặc định: `AMR3-structbart-L`
- Có sẵn: `AMR2-structbart-L`, `AMR3-structbart-L`

### 📊 Kiểm tra tiến độ

Xem thông tin progress hiện tại:

```bash
.venv/bin/python progress_utils.py --show process.json
```

Output ví dụ:
```
===== Current Progress =====
Processed:     1500
Failed:        3
Current Index: 1503
Last Update:   2024-03-12T10:45:30.123456
========================
```

### 🔄 Reset tiến độ

Nếu muốn bắt đầu lại với backup file cũ:

```bash
.venv/bin/python progress_utils.py --reset process.json
# → Tạo process.json.backup
```

### 📁 Định dạng file

#### **Input (JSONL)**
```json
{"sent": "The girl travels and visits places"}
{"sent": "A dog is running in the park"}
...
```

#### **Output (Text/Penman)**
```
# ::id id0
# ::annotator bart-amr
# ::snt The girl travels and visits places
(t / travel-01
   :ARG0 (g / girl)
   :ARG4 (v / visit-01
            :ARG0 g
            :ARG1 (p / place)))

# ::id id1
# ::annotator bart-amr
# ::snt A dog is running in the park
...
```

#### **Progress tracking (JSON)**
```json
{
  "start_time": "2024-03-12T10:00:00.123456",
  "last_update": "2024-03-12T10:45:30.654321",
  "total_processed": 1500,
  "total_failed": 3,
  "current_index": 1503,
  "processed_indices": [0, 1, 2, ..., 1502],
  "failed_indices": [150, 500, 1000],
  "errors": [
    {
      "index": 150,
      "error": "Parsing failed",
      "timestamp": "2024-03-12T10:15:30.123456"
    },
    ...
  ]
}
```

### 🎯 Scenario thực tế

**Scenario 1: Parse file lớn, GPU bị timeout **

```bash
# Chạy lần 1 - bị ngắt sau 2 giờ (processing 5000 sentence)
.venv/bin/python run.py --input data.jsonl --output output.txt --process prog.json
# ✓ Đã xử lý 5000 câu, failed 2 câu

# Chạy lần 2 - tiếp tục từ câu thứ 5000
.venv/bin/python run.py --input data.jsonl --output output.txt --process prog.json
# ✓ Tiếp tục từ câu 5000, xử lý thêm 5000 câu mới
```

**Scenario 2: Muốn xử lý với model khác**

```bash
# Lần đầu với AMR3
.venv/bin/python run.py --input data.jsonl --output amr3.txt --process prog3.json --model AMR3-structbart-L

# Lần hai với AMR2 (file progress khác)
.venv/bin/python run.py --input data.jsonl --output amr2.txt --process prog2.json --model AMR2-structbart-L
```

### ⚙️ Cấu trúc file code

```
transition-amr/
├── inference.py         # Engine chính xử lý parsing
├── run.py              # Wrapper CLI đơn giản
├── progress_utils.py   # Utilities để xem/reset progress
├── setup.sh            # Script cài đặt environment
└── PARSING_HOWTO.md    # File này
```

### 🐛 Troubleshooting

**Q: Bị error về CUDA?**
A: Kiểm tra phiên bản CUDA:
```bash
nvidia-smi
```
Và cài PyTorch phù hợp trong setup.sh

**Q: Progress.json quá lớn?**
A: File JSON chứa danh sách tất cả index đã xử lý. Nếu quá lớn, có thể reset và bắt đầu lại:
```bash
python progress_utils.py --reset process.json
```

**Q: Muốn xóa output và chạy lại?**
A: 
```bash
rm output.txt process.json
.venv/bin/python run.py --input data.jsonl --output output.txt --process process.json
```

**Q: Performance chậm?**
A: Check GPU usage:
```bash
watch -n 1 nvidia-smi
```
Có thể model lớn (structbart-large) cần GPU memory cao.

### 📞 Support

Xem chi tiết tham số:
```bash
.venv/bin/python run.py --help
.venv/bin/python inference.py --help
.venv/bin/python progress_utils.py --help
```
