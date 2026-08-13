# Waste Scanner AI v1.10.6

Ứng dụng web mở camera trên điện thoại hoặc máy tính, chụp/chọn ảnh rác, dùng CLIP để dự đoán nhóm rác và hiển thị hướng dẫn xử lý.

## Khởi động nhanh trên Windows

Khuyến nghị Python 3.11 hoặc 3.12.

Chạy bình thường:

```bat
start.bat
```

Lần đầu, `start.bat` tự tạo `.venv` và cài `requirements.txt`.

Chế độ phát triển có auto-reload:

```bat
start.bat dev
```

Chỉ cài/cập nhật dependency:

```bat
start.bat setup
```

Ứng dụng mặc định chạy tại:

```text
http://localhost:8000
```

## Chạy bằng Python trên macOS/Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python launcher.py
```

Chế độ phát triển:

```bash
python launcher.py --reload
```

Đổi cổng:

```bash
python launcher.py --port 8001
```
## Docker

Các file Docker nằm trực tiếp ở thư mục gốc của project.

Tạo `.env` từ `.env.example` nếu cần cấu hình riêng, sau đó từ thư mục gốc chạy:

```bash
docker compose --env-file .env up --build waste-scanner
```

## Chức năng chính

- Camera trước/sau trên trình duyệt.
- Chụp ảnh hoặc chọn ảnh có sẵn.
- Giảm kích thước ảnh trước khi gửi/phân loại để giảm độ trễ.
- Có 9 danh mục hướng dẫn. AI nhận diện trực tiếp 8 nhóm cụ thể: nhựa cứng/chai-hộp, nilon/nhựa mềm, giấy, kim loại, thủy tinh, hữu cơ, nguy hại và điện tử; "rác còn lại" không cạnh tranh trực tiếp với các nhóm này mà đóng vai trò danh mục dự phòng/tra cứu.
- Hiển thị mức phù hợp AI và cảnh báo khi kết quả chưa đủ chắc chắn.
- Lưu lịch sử bằng SQLite dùng chung cho mọi điện thoại/máy tính kết nối cùng máy chủ; client ID chỉ còn là metadata nguồn scan.
- Responsive cho điện thoại và máy tính.
- Hỗ trợ localhost và Docker Compose.

> Mức phù hợp AI là điểm tương đối giữa các nhãn CLIP, không phải xác suất đã được hiệu chỉnh rằng dự đoán chắc chắn đúng.