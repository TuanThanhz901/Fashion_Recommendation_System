# HỆ THỐNG KHUYẾN NGHỊ SẢN PHẨM THỜI TRANG DỰA TRÊN HÌNH ẢNH SẢN PHẨM

Hệ thống gợi ý sản phẩm là công cụ quan trọng giúp cá nhân hóa trải nghiệm mua sắm trực tuyến. Bài toán này kết hợp hành vi người dùng (lịch sử mua hàng, cho thuê đánh) và thông tin sản phẩm  như hình ảnh sản phẩm, mô tả, tên , danh mục để đề xuất các sản phẩm phù hợp. Đặc biệt trong lĩnh vực thời trang, việc phân tích hình ảnh giúp hệ thống hiểu rõ hơn về đặc điểm trực quan (màu sắc, kiểu dáng) và đưa ra gợi ý chính xác hơn.


# DỮ LIỆU SỬ DỤNG

VCR (Vibrent Clothes Rental Dataset) một bộ dữ liệu công khai trên kaggle với hơn 64.4k giao dịch người dùng kèm hình ảnh sản phẩm. Dataset bao gồm:
+ Danh sách outfits : Các bộ trang phục cho thuê/bán mỗi bộ gồm n ảnh không đồng nhất kèm thông tin về sản phẩm (danh mục,tên, giá, mô tả)
+ Hành vi người dùng: Id ẩn danh và thông tin thuê/mua các outfits.
+ Danh sách hình ảnh sản phẩm: Ảnh thuộc về mỗi outfits

# CẤU TRÚC

Nội dung triển khai của dự án gồm 2 phần : 
1/ rs_img_app : là triển khai xây dựng hệ thống khuyến nghị dựa trên hình ảnh sử dụng embeddings MobileNetV2 của ảnh đã được detect với YOLOv5s,phát triển thành ứng dụng web với flask python, hosting trên nền tảng ngrok
2/ hrs_system : là triển khai và phát triển dự án HRS được nghiên cứu bởi các giảng viên của Trường Đại học TÔN ĐỨC THẮNG, được giảng viên TRẦN TRUNG TÍN hướng dẫn và phát triển dựa trên nghiên cứu ban đầu, kết hợp đặc trưng hình ảnh vào hệ thống GCN ( Graph Convolution Networks ), hệ thống thuộc HRS ( Hybrid Recommendation System ) kết hợp kiếm trúc đồ thị trong phân tích hành vi người dùng (user_embed) và đặc trưng sản phẩm (item_embed) gồm có văn bản và hình ảnh ( mục tiêu chính của triển khai ) sau đó chúng tôi tiến hành đánh giá hiệu quả của hệ thống cho 3 trường hợp: HRS cho item_embed chỉ đặc trưng văn bản , HRS cho item_embed chỉ đặc trưng hình ảnh , và HRS với phương pháp kết hợp tối ưu đánh giá trên các mô hình LightGCN, CombiGCN , BPRMF, NGCF.
  

# MÔI TRƯỜNG PHÁT TRIỂN

Dự án được phát triển và kiểm thử trên các nền tảng Cloud Computing Platforms như Google Colab, Kaggle phiên bản hợp lệ các thư viện chính được sử đụng trong dự án như sau :
- Python 3.11.11
- TensorFlow version: 2.18.0
- NumPy version: 1.26.4
- SciPy version: 1.13.1
- Scikit-learn version: 1.6.1
Đối với flask-app và thiết lập ngrok chúng tôi sử dụng các lệnh cài đặt như sau :
< !pip install -q faiss-cpu >
< !pip install --upgrade pyngrok >
< !pip install flask flask_cors > 

# HƯỚNG DẪN CHẠY

Triển khai chia làm 2 phần rs_img_app và hrs_system với mỗi phần vào thư mục tương ứng xem readme hướng dẫn chạy chi tiết.
