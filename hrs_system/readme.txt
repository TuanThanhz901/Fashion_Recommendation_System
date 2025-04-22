# Trường hợp chạy hệ thống trên local
Trước khi chạy cần đảm bảo vị trí chạy từ thư mục ./hrs_system/ và đảm bảo các phiên bản thư viện hợp lệ như đã đề cập trước đó
Trong dự án đánh giá thống nhất toàn bộ hệ thống chạy với 3 layer 64 

<!-- lệnh command chạy -->

py LightGCN.py --dataset vcr5p_late_fusion --Ks [1,5,10,20] --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 1000 

# Chạy kiểm thử trên colab
cần đảm bảo đưa toàn thư mục hrs_system lên Google drive và sử dụng mount drive truy cập vào thư mục này. chi tiết xem file chạy mẫu TextGCN.ipynb và file đánh giá hiệu suất Eval_All.ipynb

lưu ý:
- có thể đổi LightGCN.py thành các model được cung cấp trong ./hrs_system/
- có thể đổi vcr5p_late_fusion thành các fraction dataset  khác được cung cấp trong ./hrs_system/Data 