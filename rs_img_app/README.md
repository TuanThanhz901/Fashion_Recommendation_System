
## Giới thiệu
Hệ thống gợi ý sản phẩm thời trang dựa trên hình ảnh sản phẩm sử dụng faiss index trên tập dữ liệu embeddings MobileNetV2 từ các hình ảnh đã được detect với YOLOv5s

## Hướng dẫn chạy
Ứng dụng được phát triển trên nền tảng Kaggle dự án chính thức sử dụng và import dataset từ Kaggle tham khảo qua liên kết < https://www.kaggle.com/code/guutran/rs-vibrent >  
Đối với trường hợp sử dụng data gốc và chạy local cần thay đổi các đường dẫn tương ứng đến các thư mục được cung cấp ( không đổi tên thư mục )

Quy trình chạy:

[detect và embeddings hình ảnh sản phẩm]
Chạy thủ công quá trình detect phần chính của các hình ảnh từ folder image dùng nó làm đầu vào để get embeddings với MobileNetV2 ở thư mục detect_YOLOv5s thu được embeddings lưu ở thư mục data

[hệ thống gợi ý hình ảnh sản phẩm app.py]

Quy trình chạy app.py
1. Chạy lần lượt các bước từ trên xuống bao gồm các cài đặt thư viện các Class cần thiết cho hệ thống 
2. Kiểm tra trên tập dữ liệu test (0.2 random data từ 50.3k embeddings ảnh với MobileNetV2)
3. Cài đặt và cấu hình Ngrok (có thể thay bằng Auth Token của chính tài khoản của người chạy )
4. Tải lên giao diện từ testing-rs-system nằm trong data hoặc templates 
5. Ở bước running trên cổng port 5000, click vào public url mà ngrok đã tạo liên kết và nhấn vào visit site để chuyển đến trang web chính thức 
6. Testing với hình ảnh sản phẩm tùy chỉnh (đánh giá trong dự án sử dụng hình ảnh ở testing-rs-system ) và chọn top k , hệ thống sẽ gợi ý cho trường hợp gợi ý trên sản phẩm mới và trả về kết quả tìm kiếm  k ảnh tương tự nhất từ dữ liệu ảnh được học tập.


