import os
import random  
import spacy
from spacy.util import minibatch, compounding
import  matplotlib.pyplot as plt
import numpy as np

# SỬ DỤNG BỘ PHÂN LOẠI HỌC MÁY DỂ DỰ ĐOÁN CẢM XÚC 
# Xây dựng cấu trúc thư mục của dl, tìm kiếm và mở các tệp văn bản,
# sau đó thêm 1 loạ nội dung và 1 từ điển nhãn và list reviews

def load_training_data(
	data_directory: str = "aclImdb/train",
	split: float = 0.8,
	limit: int = 0
) -> tuple: # hàm load_training_data có kiểu trả về dạng tuplr

	# load from files / tải các 
	reviews = []
	for label in ["pos", "neg"]:
		labeled_directory = f"{data_directory}/{label}" # lấy đường 
		for review in os.listdir(labeled_directory): # trả về 1 danh sách chưuas tên của các mục trong thu mục được cung cấp bởi đường 
			if review.endswith(".txt"): # kiểm tra chuỗi có kết thúc bằng ".txt"
				with open(f"{labeled_directory}/{review}", encoding="utf-8") as f: # mở tệp và gán nội dung tệp vào f. tệp tự động đóng khi câu  rời khỏi khối with
					text = f.read() # đọc toàn bộ tệp f, lưu nd vb vào 
					text = text.replace("<br />", "\n\n") # thay thế các thẻ <br /> html trong văn bản thành \n\n
					if text.strip(): # .strip() => xóa tất cả khoảng trắng đầu và cuối
						spacy_label = {
							"cats" : {
								"pos" : "pos" == label,
								"neg" : "neg" == label,
							}
						}
						reviews.append((text, spacy_label))
	# xáo trộn các tệp
	random.shuffle(reviews)

	if limit: # nêu limit != 0 
		reviews = reviews[:limit] # cắt bớt dl
	split = int(len(reviews) * split) # điểm chia traning set và test set
	return reviews[:split], reviews[split:]


# def f(ham: str, eggs: str = 'eggs') -> str:
#     print("Annotations:", f.__annotations__)
#     print("Arguments:", ham, eggs)
#     return ham + ' and ' + eggs

# print(f("a", "apple"))
# print(f("a"))
# Đó là những gợi ý về kiểu. 
# Nhiều người kiểm tra loại khác nhau có thể sử dụng chúng để xác định xem bạn có đang sử dụng đúng loại hay không. 
# Trong ví dụ tr, hàm đang mong đợi ham loại str và eggs loại str(mặc định là eggs). 
# Cuối cùng -> str ngụ ý rằng hàm này, cũng phải có kiểu trả về str.


# DL được lưu trong load_training_data() và thành phàn trong textcat

# Trước khi đi vào đào tạo, t cần tắt các thành phần đường ống khác ngoại trừ textcat.
# Điều này là để ngăn các thành phần khác bị ảnh hưởng khi đào tạo.
# Thực hiện điều này thông qua phương thức disable_pipes().
# Tiếp, sử dụng func begin_training() sẽ trả về 1 trình tối ưu (optimizer).

# t có thể sửa số lần lăp lại mô hình bằng cách thay đổi giá trị tham số iterations.
# tuy nhiên nó phải tối ưu.
# Trong mỗi lần lặp, t sẽ lặp lại các training_data và phân chia chúng thành các lô (batch)
# bằng cách sử dụng các trình trợ giúp minibatch và compounding của spaCy.

# func minibatch() của spaCy sẽ trả về các lô training data. 
# nó nhận tham số size làm kích thước của lô. sử dụng func compuding() để tạo ra chuỗi giá trị kép vô hạn.

# Với mỗi vòng lặp, mô hình sẽ được update thông qua  nlp.update().

# Việc đào tạo đã hoàn tất. t có thể đánh giá các dự đoán được thực hiện bởi mô hình 
# bằng cáhc gọi hàm evaluate_model() 
# ĐÀO TẠO TRÌNH PHÂN LOẠI
# 1. Sửa đổi spaCy Pipleline to Bao gồm textcat
def train_model(
	training_data: list,
	test_data: list,
	iterations: int = 20 # sự lặp 
) -> None:
	# BUILD PIPELINE
	nlp = spacy.load("en_core_web_sm")
	if "textcat" not in nlp.pipe_names: # ['tagger', 'parser', 'ner']
		textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"}) # Tạo một thành phần đường ống từ một factory.
		nlp.add_pipe(textcat, last=True)

		# nlp = spacy.load("en_core_web_sm")
		# print(nlp.pipe_names) 			#['tagger', 'parser', 'ner']
		# textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"})
		# nlp.add_pipe(textcat, last=True) 	#['tagger', 'parser', 'ner', 'textcat']
		# print(nlp.pipe_names)
	else:
		textcat = nlp.get_pipe("textcat")

	# Thêm nhãn với textcat để nó biết những gì cần tìm
	textcat.add_label("pos")
	textcat.add_label("neg")

	# TRAIN ONLY TEXTCAT: Xây dựng vòng lặp đào tào để tào tạo textcat
	# ds tất cả thành phần trong pipeline k phải textcat
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
	# sử dụng trình nlp.disable() quản lý ngữ cảnh để tắt các thành phần đó
	# cho tất cả mã trong phạm vi của trình quản lý ngữ cảnh
	with nlp.disable_pipes(other_pipes): 
		# hàm begin_training() trả về hàm tối ưu hóa ban đầu. 
		# đây là những gì nlp.update() sẽ sử dụgn để cập nhập trọng số của mô hình cơ bản
		optimizer = nlp.begin_training() 
		# Khởi tạo đường ống để đào tạo và trả về một Optimizer (trình tối ưu )

		# training loop
		print("Beginning training")
		print("{:<10}\t{:<25}\t{:<25}\t{:<25}\t{:<25}".format("Iteration", 
																"Loss", 
																"Precision", 
																"Recall", 
																"F-core"))
		
		loss_arr = []
		precision_arr = []
		recall_arr = []
		f_score_arr = []
		batch_sizes = compounding(4.0, 32.0, 1.001)  # return về 1 generator 
		# - Mang lại chuỗi giá trị kép vô hạn. 
		#   Mỗi lần gọi trình tạo, một giá trị được tạo ra bằng cách nhân giá trị trước đó với tỷ lệ kép. 
		#   nó sẽ được minibatch() sủ dụng sau này.
		# batch_size = next(batch_sizes)  # 4.0
		# batch_size = next(batch_sizes)  # 4.0 * 1.001

		# performing 
		for i in range(iterations):
			loss = {}
			random.shuffle(training_data)
			batches =  minibatch(training_data, size=batch_sizes)
			# Tăng kích thước lo mỗi lần update, k phải mỗi epoch.
			# compounding rate sẽ dẫn tới việc tăng rất chậm nếu chúng ta thực hiện tăng mỗi epoch.
			# Việc  tăng kích thước lô chỉ là để giúp mô hình bắt đầu, do việc học bắt chước (imitation learning).
			# imitation learning nghĩa là mục tiêu không cố định, chưa xác định trước,
			# vì vậy nếu chúng ta sử dụng batch size lớn khi bắt đầu training,
			# nó sẽ khiến máy dành quá nhiều thời gian để học  

			for batch in batches:
				text, labels = zip(*batch) 
				# zip => nén một loạt các đối số (argument) lại với nhau
				# * trong một hàm gọi là "unpacks" (giải nén) một danh sách (hoặc có thể lặp lại khác),
				# làm cho mỗi phần tử của nó trở thành 1 đối số / argument riêng biệt
				# >>> list = [[1,2,3],[4,5,6]] 
				# >>> zip(list) 	# ~ zip([[1,2,3],[4,5,6]]) 	=> ([1, 2, 3],), ([4, 5, 6],)
				# >>> zip(*list) 	# ~ zip([1,2,3], [4,5,6]) 	=> [(1, 4), (2, 5), (3, 6)]

				nlp.update(
					text, # batch of texts
					labels, # batch of annotations
					drop=0.2, 
					sgd=optimizer, 
					losses=loss
				) # training thực tế trên mỗi vd/batch		

				# Ở mỗi từ, nó đưa ra một dự đoán. 
				# Sau đó, nó tham khảo các label để kiểm tra xem dự đoán có đúng hay không. 
				# Nếu không, nó sẽ điều chỉnh trọng sô để lần sau hành động đúng sẽ đạt điểm cao hơn.
				# drop: Điều này thể hiện tỷ lệ dropout.
				# losses: Một từ điển để lưu giữ các tổn thất đối với từng thành phần đường ống. Tạo một từ điển trống và chuyển nó vào đây.
				# sgd: you have to pass the optimizer

			# Calling the evaluate_model() function and printing the scores
			with textcat.model.use_params(optimizer.averages):
				evaluation_results = evaluate_model(
					tokenizer = nlp.tokenizer,
					textcat = textcat,
					test_data = test_data
				)
				print("{:<10d}\t{:<25.15f}\t{:<25.15f}\t{:<25.15f}\t{:<25.15f}".format(i+1, 
																			loss['textcat'], 
																			evaluation_results['precision'], 
																			evaluation_results['recall'], 
																			evaluation_results['f_score'])
																			)
				# lưu lại các giá trị dùng cho việc vẽ biểu đồ bên dưới
				loss_arr.append(loss['textcat'])
				precision_arr.append(evaluation_results['precision'])
				recall_arr.append(evaluation_results['recall'])
				f_score_arr.append(evaluation_results['f_score'])
	
	# SAVE MODEL: lưu mô hình được đào tạo vào thư mục model_artifacts nằm trong thư mục hiện đang làm việc,
	# nhằm mục đích có thể sử dụng lại mà k cần đào tạo mô hình mới.
	with nlp.use_params(optimizer.averages):
		nlp.to_disk("model_artifacts")

	# BIỂU ĐỒ CHO THẤY HIỆU SUẤT CỦA MÔ HÌNH QUA 20 LẦN LẶP LẠI ĐÀO TẠO.
	# 1. biểu đồ cho thấy sự mất mát thay đổi ntn trong quá trình đào tạo
	x = np.arange(0, 20, dtype=int)
	plt.plot(x, loss_arr, 'b-', label="Loss")
	plt.legend(loc = 'upper right')
	plt.show()

	# 2. biểu dồ biểu thị độ chính xác, recall, và f-score
	plt.plot(x, precision_arr, 'b-', label="Precision")
	plt.plot(x, recall_arr, '-', label="Recall")
	plt.plot(x, f_score_arr, 'r-', label="F-Score")
	plt.legend(loc = 'upper right')
	plt.show()


# trước đó t đã chuẩn bị dl đào tạo ở dạng mong muốn và lưu trữ nó trong train_data.
# Ngoài ra, t đã có thành phần phân loại văn bản của mô hình trong textcat.
# Vì vậy,t có thể tiến hành đào tạo textcat trên tập train_data.
# Nhưng mà, k phải chúng ta dang thiếu thứ gì đó sao?

# Đối với bất kỳ mô hình nào mà t đào tạo, điều quan trong là phải kiếm tra xem nó có đúng như mong đợi của t hay k.
# Đây được gọi là đánh giá mô hình. Đây là bước k bắt buộc nhưng rất đượcc khuyến khíhc để có kq tốt hơn.
# Nếu t nhớ lại, hàm load_data() đã phân chia khoảng 20% dl gốc để đánh giá. 
# T sẽ sd những dl này để kiểm tra mức độ tốt của khóa đào tạo.

# Vì v, t đã viết 1 hàm evaluate_model() để thực hiện quá trình đánh giá mô hình.
# T sẽ gọi hàm này sau trong quá trình đào tạo để xem hiệu suất của mô hình.

# Hàm này sẽ lấy textcat và dl đánh giá làm đầu vào.
# Đối với mỗi text trong dl đánh giá, nó đọc score từ các dự đoán được thực hiện.
# Và dựa trên điều này, nó sẽ tính toán các giá trị True +, True -, Falase +, False -.

# True positive (tp):
# False positive (fp):
# True negative (tn):
# True negative (fn):

# Một mô hình tốt phải có độ chính xác (prescision) ttoots cũng như recall cao.
# Vì vậy, lý tưởng nhất là t muốn có 1 thước đo kết hợp cả 2 khía cạnh này trong 1 chỉ số duy nhất - f_score
# F-Score = (2 * Precision * Recall) / (Precision + Recall)

# 2. ĐÁNH GIÁ TIẾN ĐỘ ĐÀO TẠO MÔ HÌNH
def evaluate_model(
	tokenizer, textcat, test_data: list
) -> dict:
	# tách từng bài đánh giá và nhãn trong test_data
	reviews, labels = zip(*test_data) 
	reviews = (tokenizer(review) for review in reviews)

	# dùng biểu thức trình tạo mã hóa từng bài đánh giá, 
	# chuẩn bị cho chúng được chuyển đến textcat
	# biểu thức trình tạo là 1 thủ thuật hay được đề xuất 
	# trong tài liệu spaCy cho phép lặp lại các bài đánh giá
	# được mã hóa mà không cần lưu từng bài đánh giá trong bộ nhớ.

	true_positives = 0.0 
	false_positives = 1e-8 # k thể là 0 vì sự hiện diện ở mẫu 
	true_negatives = 0.0
	false_negatives = 1e-8

	for i, review in enumerate(textcat.pipe(reviews)):
		true_label = labels[i]['cats']
		for predicted_label, score in review.cats.items(): 
			# mọi danh mục từ điển bao gồm cả hai nhãn. 
			# bạn có thể nhận được tất cả thông tin bạn cần chỉ với nhãn tích cực
			if predicted_label == "neg":
				continue
			if score >= 0.5 and true_label["pos"]:
				true_positives += 1
			elif score >= 0.5 and true_label["neg"]:
				false_positives += 1
			elif score < 0.5 and true_label["neg"]:
				true_negatives += 1
			elif score < 0.5 and true_label["pos"]:
				false_negatives += 1

	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)

	if precision + recall == 0:
		f_score = 0
	else:
		f_score = (2 * precision * recall) / (precision + recall)
	return {"precision": precision, "recall": recall, "f_score": f_score}


TEST_REVIEW = """Transcendently beautiful in moments outside the office, 
it seems almost sitcom-like in those scenes. 
When Toni Colette walks out and ponders life silently, it's gorgeous.<br /><br />
The movie doesn't seem to decide whether it's slapstick, farce, magical realism, or drama, nut 
the best of it doesn't matter. 
(The worst is sort of tedious - like Office Space with less humor.)"""

# 3. PHÂN LOẠI ĐÁNH GIÁ 
def test_model(input_data: str=TEST_REVIEW):

	# Load saved trained model: Tải mô hình đã lưu trước đó
	loaded_model = spacy.load("model_artifacts")

	# tạo dự đoán
	parsed_text = loaded_model(input_data)

	# xác định kết quả dự đoán
	if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
		prediction = "Positive"
		score = parsed_text.cats["pos"]
	else:
		prediction = "Negative"
		score = parsed_text.cats["neg"]

	# in kết quả dự đoán
	print(
		f"Review text: {input_data}\nPredicted sentiment: {prediction}"
		f"\tScore: {score}"
	)


# 4. KẾT NỐI
if __name__ == "__main__":
	# tải training set và test set, 
	# giới hạn tổng số bài đánh giá được sử dụng là 2500
	train, test = load_training_data(limit=2500) 
	train_model(train, test) # tạo mô hình
	print("Testing model") 
	test_model() # gọi test_model() để kiểm tra hiệu suất của mô hình

	print("----------------------------------------------------------")
