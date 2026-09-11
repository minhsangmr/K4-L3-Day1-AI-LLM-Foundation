# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, phản hồi khá ổn định, trực tiếp và ít thay đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, câu trả lời bắt đầu đa dạng hơn, có thêm cách diễn đạt hoặc ví dụ khác nhau. Ở temperature = 1.5, phản hồi thường sáng tạo hơn nhưng cũng có nguy cơ lan man hoặc kém nhất quán hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, tôi sẽ đặt temperature khoảng 0.2 hoặc 0.3. Lý do là chatbot cần trả lời nhất quán, chính xác và tuân thủ chính sách hỗ trợ hơn là sáng tạo. Temperature thấp giúp giảm nguy cơ trả lời lan man hoặc tạo thông tin không đáng tin cậy.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload này tạo khoảng 10.500.000 token đầu ra mỗi ngày. Theo bảng giá trong bài, riêng chi phí output của GPT-4o là khoảng 105 USD/ngày, còn GPT-4o-mini là khoảng 6.3 USD/ngày, tức GPT-4o đắt hơn khoảng 16.7 lần. GPT-4o xứng đáng dùng cho các tác vụ khó như tư vấn chuyên sâu, phân tích tài liệu quan trọng hoặc yêu cầu lập luận tốt; còn GPT-4o-mini phù hợp cho chatbot FAQ, phân loại nội dung đơn giản hoặc trả lời các câu hỏi phổ biến với chi phí thấp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, phản hồi thường ngắn hơn, dùng từ đơn giản và ví dụ gần gũi như một cuốn sổ ghi chép được nhiều người cùng kiểm tra. Với persona chuyên gia tài chính, phản hồi dài và chuyên sâu hơn, dùng nhiều thuật ngữ như sổ cái phân tán, cơ chế đồng thuận, tài sản số, giao dịch và rủi ro. Điều này cho thấy system prompt có ảnh hưởng mạnh đến vai trò, mức độ chi tiết, cách chọn từ vựng và loại ví dụ mà model sử dụng. Cùng một câu hỏi nhưng model có thể điều chỉnh câu trả lời theo đối tượng người nghe.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi chọn một đoạn tiếng Việt khoảng 100 từ. Cách ước lượng số từ / 0.75 cho ra khoảng 133 token, trong khi count_tokens bằng tiktoken cho ra khoảng 160–180 token, tức cao hơn khoảng 20–35% tùy đoạn văn. Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì có dấu, nhiều âm tiết tách bằng khoảng trắng, và tokenizer thường được tối ưu tốt hơn cho tiếng Anh. Vì vậy đếm từ chỉ là ước lượng thô, còn tiktoken phản ánh gần hơn cách model thật xử lý văn bản.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi có thể dài hoặc mất nhiều thời gian sinh, ví dụ chatbot, trợ lý lập trình, giải thích bài học hoặc viết nội dung dài. Người dùng thấy từng phần câu trả lời xuất hiện ngay nên cảm giác hệ thống nhanh và tự nhiên hơn, dù tổng thời gian sinh có thể không giảm nhiều. Non-streaming phù hợp hơn khi phản hồi ngắn, hoặc khi ứng dụng cần nhận toàn bộ kết quả trước để kiểm tra định dạng JSON, kiểm duyệt nội dung, tính toán thêm hoặc lưu vào hệ thống. Vì vậy streaming tối ưu trải nghiệm tương tác, còn non-streaming đơn giản và dễ kiểm soát hơn trong pipeline xử lý tự động.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff có lợi vì sau mỗi lần thất bại, client chờ lâu hơn trước khi thử lại, giúp giảm áp lực lên API khi hệ thống đang quá tải. Nếu luôn dùng delay cố định như 1 giây, hàng nghìn client có thể cùng retry gần như cùng thời điểm, tạo ra một đợt request mới làm server tiếp tục quá tải. Hiện tượng này giống “thundering herd”, khiến lỗi kéo dài hơn thay vì phục hồi. Backoff theo cấp số nhân phân tán request theo thời gian và cho server thêm cơ hội ổn định lại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là trợ giảng AI thân thiện cho sinh viên mới học LLM API. System prompt của tôi là: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, giải thích từng bước và luôn đưa ví dụ đơn giản khi cần.” Tôi dùng cụm “trả lời ngắn gọn” để tránh câu trả lời quá dài trong giao diện CLI và giúp người học dễ theo dõi. Tôi cũng chỉ định “bằng tiếng Việt” vì người dùng mục tiêu là sinh viên Việt Nam trong lab này, còn “từng bước” giúp model ưu tiên cách giải thích có cấu trúc.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là history chỉ giữ 3 lượt gần nhất, nên model dễ quên các thông tin quan trọng đã xuất hiện sớm trong cuộc trò chuyện. Một cải thiện cụ thể là thêm bộ nhớ tóm tắt dài hạn: sau mỗi vài lượt, hệ thống dùng model hoặc hàm riêng để tóm tắt các thông tin quan trọng rồi lưu vào một biến summary. Khi gửi request mới, messages sẽ gồm system prompt, summary và 3 lượt hội thoại gần nhất. Cách này giúp giữ ngữ cảnh quan trọng mà không làm số token tăng quá nhanh.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
