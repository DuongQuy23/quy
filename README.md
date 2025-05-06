Bài 2: Dự đoán giá trị liên tục
Mô tả: Sử dụng tập dữ liệu "Boston Housing" từ sklearn.datasets hoặc Kaggle, nhưng mở rộng bài tập với các kỹ thuật học máy nâng cao, xử lý đặc trưng phức tạp, và đánh giá mô hình chuyên sâu.
Yêu cầu:
+ Vẽ pairplot để kiểm tra mối quan hệ giữa các đặc trưng và biến mục tiêu (price).
+ Sử dụng kiểm định thống kê (ví dụ: Pearson correlation) để xác định các đặc trưng có mối quan hệ tuyến tính mạnh với price.
+ Xử lý giá trị ngoại lai (outliers) trong price và các đặc trưng khác bằng phương pháp Isolation Forest thay vì chỉ dùng IQR.
+ Kiểm tra và xử lý hiện tượng đa cộng tuyến (multicollinearity) giữa các đặc trưng bằng cách tính Variance Inflation Factor (VIF). Loại bỏ hoặc kết hợp các đặc trưng có VIF cao (>5).
+ Tính tỷ lệ room_per_crime (số phòng trung bình chia cho tỷ lệ tội phạm).
+ Tạo biến phân loại high_tax (1 nếu thuế suất vượt ngưỡng trung bình, 0 nếu không).
+ Tạo các đặc trưng tương tác (interaction terms) giữa các cặp đặc trưng có tương quan cao với price (ví dụ: RM * LSTAT).
+ Sử dụng Polynomial Features (độ 2) để tạo các đặc trưng phi tuyến từ các đặc trưng gốc.
+ Huấn luyện ít nhất 3 mô hình: Linear Regression, Gradient Boosting Regressor (XGBoost hoặc LightGBM), và Neural Network (sử dụng Keras hoặc TensorFlow).
+ Xây dựng một pipeline tích hợp các bước tiền xử lý (chuẩn hóa, mã hóa, xử lý ngoại lai) và mô hình học máy.
+ Sử dụng Bayesian Optimization (thư viện như scikit-optimize hoặc optuna) để tối ưu hóa siêu tham số của Gradient Boosting Regressor (ví dụ: learning_rate, n_estimators, max_depth).
+ Kết hợp các mô hình bằng kỹ thuật Stacking (sử dụng sklearn.ensemble.StackingRegressor) với Linear Regression làm meta-learner.
+ Đánh giá hiệu suất bằng cross-validation (5-fold) với các chỉ số: MSE, RMSE, R², và Mean Absolute Percentage Error (MAPE).
+ Vẽ biểu đồ residual plot để phân tích sai số của mô hình tốt nhất.
+ Thực hiện phân tích SHAP (SHapley Additive exPlanations) để giải thích đóng góp của từng đặc trưng vào dự đoán của mô hình Gradient Boosting.

Bài 5: Phân tích dữ liệu giáo dục
Mô tả: Bạn là nhà phân tích dữ liệu tại một trường đại học, được giao nhiệm vụ phân tích hiệu suất học tập của sinh viên dựa trên dữ liệu giả định: ID sinh viên, điểm các môn (toán, văn, khoa học), số giờ tự học, số buổi vắng mặt, và mức độ tham gia hoạt động ngoại khóa (thấp, trung bình, cao).
Dữ liệu: https://drive.google.com/file/d/1djzdxesK8onGTdHEMNSyL4OB6WTsHyBb/view?usp=sharing
+ Tạo chỉ số "hiệu suất học tập tổng hợp" (tự định nghĩa công thức, ví dụ: kết hợp điểm môn và số giờ tự học).
+ Sử dụng kiểm định thống kê (ví dụ: ANOVA) để kiểm tra xem mức độ tham gia ngoại khóa có ảnh hưởng đáng kể đến hiệu suất học tập hay không.
+ Tạo đặc trưng "cân bằng học tập" (balanced learning) dựa trên sự chênh lệch điểm giữa các môn (ví dụ: chênh lệch thấp = cân bằng tốt).
+ Tạo đặc trưng "rủi ro học tập" dựa trên số buổi vắng mặt và số giờ tự học (tự định nghĩa ngưỡng).
+ Xây dựng mô hình SVM (Support Vector Machine) để phân loại sinh viên thành 2 nhóm: "có nguy cơ trượt" hoặc "an toàn" dựa trên đặc trưng đã tạo.
+ Tự điều chỉnh siêu tham số của SVM (như C, kernel) bằng cách thử nghiệm và ghi lại kết quả.

Bài 7: Phân tích dữ liệu y tế
Mô tả: Bạn làm việc tại một bệnh viện và được giao phân tích dữ liệu giả định về bệnh nhân tiểu đường. Dữ liệu bao gồm: tuổi, BMI, mức đường huyết, số lần nhập viện, và liệu bệnh nhân có biến chứng (có/không).
Yêu cầu: https://drive.google.com/file/d/1K3y9WnaMPi_oMFylAELFnNXtgLhFnPU4/view?usp=sharing
+ Tạo chỉ số "nguy cơ biến chứng" (tự định nghĩa, ví dụ: kết hợp BMI, đường huyết, và số lần nhập viện).
+ Sử dụng kiểm định chi-squared để kiểm tra xem biến chứng có phụ thuộc vào nhóm tuổi (phân nhóm: <40, 40-60, >60) hay không.
+ Tạo đặc trưng "xu hướng đường huyết" (tăng, giảm, ổn định) dựa trên giả định dữ liệu lịch sử (tự mô phỏng).
+ Tạo đặc trưng "mức độ nghiêm trọng" dựa trên số lần nhập viện và mức đường huyết.
+ Xây dựng mô hình Logistic Regression và Random Forest để dự đoán khả năng biến chứng.
+ Tự điều chỉnh siêu tham số của Random Forest (như n_estimators, max_depth) bằng cách thử nghiệm và so sánh hiệu suất.
+ Sử dụng kỹ thuật SMOTE để xử lý mất cân bằng dữ liệu (nếu tỷ lệ biến chứng thấp).
