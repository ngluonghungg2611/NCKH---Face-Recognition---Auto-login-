# NCKH---Face-Recognition---Auto-login-
**1. Bước đầu tiền là chúng ta hãy install git và config với ssh nhé =)) !** <br>
- Install git: https://www.youtube.com/watch?v=2j7fD92g-gE
- Setup SHH for github: https://www.youtube.com/watch?v=3aKda-oXWc8


**2. Bước đầu tiên chúng ta cần phải tải thư viện để đáp ứng được nhu cầu của chương trình**
	
	pip install -r requirements.txt

**3. Sau khi tải xong thư viện thì chúng ta bắt đầu tiền hành thu thập dữ liệu** <br>
Có 2 cách để thu thập dữ liệu trong chương trình này <br>

_**Cách 1:**_ Thu thập bằng camera: <br> <br>
<i>Chạy dòng lệnh dưới đây trong terminal</i>

	python Collecting_faces.py
	
<i>Nhập tên</i>
	
	Import your name: test 
	
<i> Sau khi nhập tên sẽ xuât hiện của sổ Camera -> Thực hiện theo đúng yêu cầu hiện trên của sổ </i>
	
<img src = "https://github.com/ngluonghungg2611/NCKH---Face-Recognition---Auto-login-/blob/master/Camera.jpg" width = 70%>

<i>Sau khi khi thực hiện bước trên, ta lưu 5 ảnh với các góc cạnh tương ứng như yêu cầu và lưu vào data faces </i>

<img src = "https://github.com/ngluonghungg2611/NCKH---Face-Recognition---Auto-login-/blob/master/file_faces.jpg">

_**Cách 1:**_ Thu thập bằng WEB local khi đã chuẩn bị sẵn hình ảnh: <br> <br>	
<i> Chạy dòng lệnh trong terminal: Run server - Sử dụng flask API để thu thập ảnh </i>

	python server.py

<i>Nhấn vào link localhost ta nhận được 1 giao diện upload ảnh cơ bản: </i>

<img src="https://github.com/ngluonghungg2611/NCKH---Face-Recognition---Auto-login-/blob/master/template_upload_files.jpg">
	
<i>Sau khi upload ảnh xong thì tương tự thu được data trong file images/</i>

**4. Sau khi có dữ liệu thì tiến hành train model bằng cách train lại pretrained-model**

	python faces_train.py
	
<i>Sau khi train xong thì model được lưu ở recognizers/faces_trainner.yml</i>

<img src="https://github.com/ngluonghungg2611/NCKH---Face-Recognition---Auto-login-/blob/master/file_trainner.jpg">


**5. Test model đã được train**

<i>Model sẽ train và thu thập các face labels vào file face-labels.pickle</i>

<img src="https://github.com/ngluonghungg2611/NCKH---Face-Recognition---Auto-login-/blob/master/face_labels.jpg">

<i>Test model đã được train bằng Camera </i>

	python faces.py

<img src="https://github.com/ngluonghungg2611/NCKH---Face-Recognition---Auto-login-/blob/master/test_model.jpg">

**Well Done! Have a good day** 
	
	
		
	
	
