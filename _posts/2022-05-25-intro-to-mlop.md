---
title: Introducing to MLOps
tags: [MLOps]
style: fill
color: secondary
comments: true
description: Introducing to MLOps.
---


# Part I. MLOps: What and Why
## 1. Why NOW and Challenge
### 1.1. Defining MLOPs and Its Challenges
MLOps là tập hợp các công cụ, phương pháp để chuẩn hóa và đơn giản hóa việc quản lý vòng đời phát triển mô hình Machine Learning (ML). 

Có ba thách thức chính trong việc quản lý vòng đời phát triển mô hình ML ở quy mô lớn:
- __*There are many dependencies*__: Trong thực tế không chỉ dữ liệu liên tục thay đổi và nhu cầu kinh doanh cũng thay đổi theo. Kết quả của mô hình liên tục được gửi lại cho doanh nghiệp để đảm bảo rằng thực tế của mô hình và dữ liệu trong môi trường Production phù hợp với kỳ vọng và giải quyết, đáp ứng được nhu cầu của doanh nghiệp. "Code is relatively static, data is always changing", dẫn đến ML models bắt buộc phải học và đáp ứng với sự thay đổi của dữ liệu.
- __*Not everyone speaks the same language*__: ML lifecycle có sự tham gia của rất nhiều người ở các vai trò khác nhau (nhóm kinh doanh, nhóm DA, nhóm DS, nhóm DE, nhóm Dev,...), nhưng hầu như không có nhóm nào trong số này đang sử dụng các công cụ giống nhau, thậm chí trong nhiều trường hợp không đủ kỹ năng cơ bản để làm cơ sở giao tiếp (VD: Nhóm kinh doanh không hiểu model, hệ thống,...).
- __*Data Scientist are not Software Engineers*__: Hầu hết các DS đều chuyên về xây dựng và đánh giá, không nhất thiết phải là chuyên gia phát triển phần mềm. Mặc dù điều này có thể thay đổi trong tương lai khi mà các DS dần nắm vững hơn về mặt triển khai và giám sát mô hình.

### 1.2. MLOps to Mitigate Risk
MLOps cho rằng việc giảm thiểu rủi ro là hết sức quan trọng khi một nhóm tập trung trong doanh nghiệp phải vận hành nhiều mô hình ML. 

Việc đẩy các mô hình ML lên Production mà không có kiến trúc MLOps là rủi ro vì nhiều lý do, một trong số đó là việc đánh giá "_đầy đủ_" hiệu suất của mô hình học máy chỉ có thể thực hiện trong môi trường Production. Nguyên nhân chủ yếu bởi vì các mô hình dự đoán chỉ tốt trên tập dữ liệu huấn luyện, có nghĩa là dữ liệu huấn luyện phải phản ánh tốt dữ liệu gặp phải trong môi trường Production (iid assumption). 

Một yếu tố rủi ro chính khác là hiệu suất của mô hình học máy thường rất nhạy cảm với môi trường Production mà nó đang chạy (bao gồm phiên bản phần mềm, OS được sử dụng). Chúng không có xu hướng lỗi theo nghĩa phần mềm cổ điển, bởi hầu hết không được viết bằng tay mà xây dựng trên một đống các thư viện, phần mềm nguồn mở. Chính vì thế việc khớp các phiên bản phần mềm giữa môi trường Dev và Production là cực kỳ quan trọng. 

Cuối cùng, việc đưa các mô hình vào môi trường Production không phải là bước cuối cùng của vòng đời phát triển mô hình ML mà thường chỉ là bước đầu của việc theo dõi hiệu suất của nó và đảm bảo rằng nó hoạt động như mong đợi. 

Khi nhiều nhà khoa học dữ liệu bắt đầu đẩy nhiều mô hình học máy hơn lên Production, MLOps trở nên quan trọng trong việc giảm thiểu rủi ro tiềm ẩn mà mô hình có thể ảnh hưởng xấu (hoặc thậm chí tàn phá) doanh nghiệp nếu như mọi thứ diễn ra không như ý muốn. Việc giám sát cũng rất cần thiết để tổ chức chính xác về mức độ sử dụng rộng rãi của từng mô hình.

### 1.3. MLOps for Responsible AI
Việc sử dụng ML có trách nhiệm bao gồm hai khía cạnh chính:
- ***Intentionality***: Đảm bảo rằng các mô hình được thiết kế và hoạt động theo những cách phù hợp với mục đích của chúng. Điều này bao gồm đảm bảo rằng dữ liệu được sử dụng đến từ các nguồn dữ liệu hợp pháp và không thiên vị. Tính *Intentionality* cũng bao gồm khả năng giải thích, có nghĩa là kết quả của các hệ thống AI phải được giải thích bởi con người (lý tưởng nhất là không chỉ con người tạo ra hệ thống).

- ***Accountability***: Accountability là khả năng tổng quát hóa việc các nhóm đang sử dụng dữ liệu nào, cách thức và mô hình nào. Nó cũng bao gồm nhu cầu tin tưởng rằng dữ liệu là đáng tín cậy và được thu thập theo quy định được ban hành. 

Responsible AI sẽ thay đổi nền tảng cơ bản của việc giải trình từ cấp thấp nhất lên cấp cao nhất. Điều này có nghĩa là các quyết định trước đây vốn được đưa ra bởi con người, giờ đây hiện đang đựa đưa ra bởi mô hình. Người chịu trách nghiệm về các quyết định tự động của mô hình có thể là người quản lý nhóm phát triển mô hình hoặc thậm chí là giám đốc điều hành.

### 1.4. MLOps for Scale 
Bên cạnh 2 lợi ích trên của MLOps, nó cũng là một thành phần thiết yếu để triển khai đại trà các mô hình học máy, từ một hoặc một số mô hình trong môi trường Production đến hàng trăm, hàng nghìn mô hình có tác động tích cực đến kinh doanh đòi hỏi phải tuân thủ MLOps

Các phương pháp MLOps tốt sẽ giúp các nhóm khoa học dữ liệu trong việc giảm thiểu các công đoạn sau:
- Theo dõi phiên bản, đặc biệt là với các thử nghiệm trong giai đoạn phát triển
- Hiểu liệu các mô hình được đào tạo lại có tốt hơn các phiên bản trước đó hay không (sau đó đẩy các mô hình có hiệu suất tốt hơn lên Production)
- Đảm bảo (tại các khoảng thời gian nhất định hàng giờ, hàng ngày, hàng tháng, etc.) hiệu suất của mô hình không bị suy giảm trong quá trình Production

## 2. Key MLOps Features
### 2.1. Model Development
#### 2.1.1. Establishing Business Objectives
Quá trình phát triển các mô hình ML thường bắt đầu với mục tiêu nghiệp vụ (VD: mục tiêu giảm các giao dịch gian lận xuống <0.1% hoặc mở rộng tập khách hàng mới lên 1%). Mục tiêu nghiệp vụ đương nhiên sẽ đi kèm với mục tiêu hoạt động, yêu cầu cơ sở hạ tầng kỹ thuật và các ràng buộc về chi phí; tất cả các yếu tố này có thể được ghi nhận bằng KPIs, điều này yêu cầu giám sát hoạt động nghiệp vụ của các mô hình trong Production.

Các dự án ML là một phần của các dự án lớn hơn, tác động đến công nghệ, quy trình và con người. Điều đó có nghĩa là một phần của việc thiết lập mục tiêu cũng bao gồm quản lý sự thay đổi, thậm chí cung cấp một số hướng dẫn về cách thức xây dựng mô hình ML. Ví dụ: mức độ minh bạch sẽ ảnh hưởng đến việc lựa chọn thuật toán và có thể thúc đẩy nhu cầu cung cấp các giải thích (Explainable AI) cùng với dự đoán để các dự đoán được biến thành các quyết định có giá trị ở cấp độ kinh doanh. 

#### 2.1.2. Data Sources and Exploratory Data Analysis
Khi mục tiêu nghiệp vụ được xác định rõ ràng, các DS sẽ tiến hành phát triển mô hình ML. Bước đầu tiên trong quá trình phát triển mô hình ML đó là việc tìm kiếm dữ liệu đầu vào phù hợp.

Việc tìm kiếm dữ liệu đầu vào phụ thuộc vào việc trả lời các câu hỏi sau:
- Tại tổ chức hoặc trên thế giới liệu đã có tập dữ liệu nào tương tự chưa?
- Tập dữ liệu hiện có liệu có đủ chính xác và đáng tin cậy không?
- Làm thế nào để các bên liên quan có thể truy cập vào bộ dữ liệu này
- Những features có thể tạo nên bằng cách kết hợp từ nhiều nguồn dữ liệu
- Tập dữ liệu này liệu có đáp ứng trong thời gian thực? Tần suất cập nhật là bao nhiêu?
- Tập dữ liệu dự đoán có cần phải gán nhãn hay sẽ sử dụng cho việc học không giám sát (unspervised learning). Nếu cần gán nhãn, điều này sẽ tốn bao nhiêu chi phí về thời gian và nguồn lực
- Data platform nào nên được sử dụng?
- Dữ liệu sẽ được cập nhật như thế nào khi mô hình được triển khai?
- Liệu việc sử dụng mô hình ML có làm giảm tính đại diện của dữ liệu không?
- Các KPIs được thiết lập 

Bên cạnh các câu hỏi trên, ta cần quan tâm thêm về vấn đề quản trị dữ liệu:
- Dữ liệu được chọn có thể sử dụng cho mục đích này hay không?
- Điều khoản sử dụng dữ liệu là gì?
- Các thông tin nhận dạng cá nhân (Personally identifiable information) có phải mã hóa hoặc ẩn danh hay không?
- Những features cá nhân, chẳng hạn như giới tính, có thể sử dụng một cách hợp pháp trong bối cảnh kinh doanh hay không?
- Các mẫu dữ liệu thiểu số có được thể hiện đầy đủ rằng mô hình có hiệu suất tương đương trên mỗi nhóm hay không?

Vì dữ liệu là thành phần quan trọng, quyết định sức mạnh của các thuật toán ML, do đó trước khi thử nghiệm training mô hình, các DS cần phải có những hiểu biết về các mẫu dữ liệu. Các kỹ thuật khai phá dữ liệu (Exploratory Data Analysis - EDA) có thể giúp các xây dựng các giả thuyết về dữ liệu, xác định các yêu cầu về làm sạch dữ liệu và lựa chọn các features quan trọng. EDA có thể thực hiện bằng cách trực quan hóa hoặc sử dụng các thông kê nếu yêu cầu nghiêm ngặt hơn.

#### 2.1.3. Feature Engineering and Selection
Những tri thức thu được từ việc khai phá dữ liệu EDA sẽ mở ra các hướng để áp dụng các kỹ thuật feature engineering. Kỹ thuật feature engineering là quá trình biến đổi dữ liệu thô từ các tập dữ liệu được chọn và chuyển nó thành các "features" thể hiện tốt hơn cho mục tiêu, vấn đề cần giải quyết. "Features" là các mảng dữ liệu số có kích thước cố định, vì nó là đối tượng duy nhất mà các thuật toán ML hiểu được. Các kỹ thuật Feature Engineering là công đoạn chiếm nhiều thời gian nhất trong quá trình xây dựng và phát triển của một dự án ML, việc xây dựng và phát triển dự án ML có thể được phân thành 2 hướng tiếp cận:
- __Data Centric__: Tập trung phát triển dữ liệu training, cố định mô hình ML.
- __Model Centric__: Tập trung vào xây dựng mô hình ML, cố định dữ liệu training.

Các kỹ thuật feature engineering có thể được chia thành 4 nhóm chính sau:
- __Derivatives__: Các kỹ thuật suy luận tạo ra thông tin mới từ những thông tin hiện có.
VD: Ngày quan sát là ngày thứ mấy trong tuần?.
- __Enrichment__: Thêm những thông tin mới cho dữ liệu. 
VD: Ngày quan sát là ngày lễ, ngày giảm giá, ngày xảy ra sự kiện nào đó?
- __Encoding__: Biễu diễn cùng một thông tin dưới dạng khác nhau. 
VD: Categorical feature mang tính thứ tự thì Encode có thứ tự (Tiểu học, trung học cơ sở, trung học phổ thông -> Tiểu học = 1, trung học cơ sở = 2, trung học phổ thông = 3). Categorical feature độc lập thì encoding không có thứ tự (phim hành động, phim tình cảm, phim tâm lý -> phim tâm lý = [0, 0, 1].  
- __Combination__: Kết hợp các thông tin đã có thành thông tin mới
VD: Tỉ lệ tiêu dùng giữa 15 ngày cuối tháng và 15 ngày đầu tháng.

Ngoài ra, đối với dữ liệu phi cấu trúc như dữ liệu văn bản, âm thanh, ảnh, video,... sẽ yêu cầu các kỹ thuật feature engineering phức tạp hơn. Deep Learning trong khoảng thời gian gần đây đã cách mạng hóa lĩnh vực này bằng cách cung cấp các mô hình chuyển đổi phi cấu trúc thành dữ liệu dạng số để có thể sử dụng các thuật toán ML. Các kỹ thuật này được gọi chung là ___embeddings___, điều này ngoài ra sẽ giúp các DS thực hiện việc transfer learning vì các embeddings này có thể được sử dụng trong các miền tri thức mà họ chưa huấn luyện. 

Việc thực hiện các kỹ thuật feature engineering có thể giúp mô hình đạt đến độ chính xác cao hơn tuy nhiên nó có nhiều mặt trái, tất cả đều có thể tác động đáng kể đến chiến lược MLOp:
- Chi phí tính toán đắt đỏ (cả về hạ tầng tính toán và thời gian tính toán)
- Nhiều features hơn tương đương với nhiều đầu vào hơn, dẫn đến yêu cầu bảo trì nhiều hơn.
- Nhiều features dẫn đến mất đi một số tính ổn định.
VD: Thêm features về covid-19, có thể tốt trong giai đoạn bùng dịch tuy nhiên sẽ không tốt trong giai đoạn dịch bệnh đã ổn định.
- Một số tính năng có thể gây nên sự lo ngại về quyền riêng tư khách hàng.

Chính vì thế, việc Feature Selection (tự động) có thể giúp ích bằng cách sử dụng phương pháp heuristics để ước tính mức độ quan trọng của một số tính năng đối với hiệu suất dự đoán của mô hình. VD: người ta có thể xem xét mối tương quan các features với biến mục tiêu hoặc nhanh chóng đào tạo một mô hình đơn giản trên một tập hợp con đại diện của dữ liệu và sau đó xem xét feature nào là yếu tố dự đoán mạnh nhất.

#### 2.1.4. Model Training and Evaluation
Sau khi chuẩn bị dữ liệu bằng các kỹ thuật Feature Engineering và Selection, bước tiếp theo là huấn luyện mô hình. Quá trình huấn luyện và tối ưu hóa một mô hình ML là quá trình lặp đi lặp lại, trong đó ta sẽ thử nghiệm nhiều thuật toán khác nhau và tối ưu tham số của chúng, có thể tạo (tự động) các features khác nhau, kỹ thuật feature selection có thể được áp dụng. Trong nhiều trường hợp, khâu huấn luyện mô hình được coi là bước chuyên sâu nhất của vòng đời phát triển mô hình ML khi nói đến khả năng tính toán.

Việc theo dõi kết quả của từng thử nghiệm khi việc thử nghiệm lặp đi lặp lại trở nên ngày càng phức tạp. Không có gì khiến các DS trở nên khó chịu hơn là họ không thể tạo ra lại kết quả tốt nhất vì họ không thể nhớ cấu hình một cách chính xác. Do đó MLOps yêu cầu sử dụng các công cụ theo dõi quá trình thử nghiệm để đơn giản hóa quá trình ghi nhớ dữ liệu, feature engineering, feature selection, các tham số của mô hình cùng với chỉ số hiệu suất tương ứng. Các công cụ này cho phép các thử nghiệm được so sánh song song với nhau, từ đó làm nổi bật sự khác biệt về hiệu suất.

Bên cạnh các tiêu chí định lượng như độ chính xác mô hình hoặc sai số trung bình, việc quyết định giải pháp tốt nhất đòi hỏi xem xét cả về các tiêu chí định tính liên quan đến khả năng giải thích của thuật toán hoặc tính dễ triển khai của nó.

#### 2.1.5. Reproducibility
Mặc dù nhiều thử nghiệm có thể tồn tại trong thời gian ngắn, nhưng các phiên bản quan trọng của mô hình cần được lưu lại để có thể sử dụng sau này. Thách thức ở đây là khả năng ___tái tạo___, đây là một khái niệm quan trọng trong khoa học thực nghiệm nói chung. Mục đích của khả năng tái tạo đó là việc lưu đủ các thông tin về môi trường mà mô hình phát triển để mô hình có thể tái tạo lại với kết quả giống như trong quá trình thử nghiệm. Nếu không có khả năng tái lập, các DS khó có thể tự tin tự tái tạo lại kết quả mô hình và tệ hơn, họ khó có thể giao mô hình cho DevOps để xem liệu những gì được tạo ra trong thực nghiệm có thể tái tạo trung thực trong môi trường Production hay không. Để đảm bảo khả năng tái tạo, MLOps yêu cầu ___version control___ của tất cả các nội dung và thông số liên quan, bao gồm dữ liệu được sử dụng để huấn luyện và đánh giá mô hình, các pipelines thực hiện feature engineering, các thuật toán và bộ tham số tương ứng, ngoài ra còn các thông số về môi trường phần mềm như các phiên bản của ngôn ngữ lập trình, các thư viện sử dụng, các mã nguồn mở sử dụng,...













