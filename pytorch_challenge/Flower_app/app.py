from flask import request
from flask import jsonify
from flask import Flask, render_template
import os


app = Flask(__name__)

from inference import get_flower_name


@app.route('/', methods= ['GET', 'POST'])

def hello_world():
	if request.method == 'GET':
		flower_name=" "
		return render_template('index.html', flower= flower_name)
	if request.method == 'POST':
		flower_name=" "
		if 'file' not in request.files:
			print("file not uploaded")
			return
		
		file = request.files['file']
		#print('file uploaded')
		image = file.read()
		flower_name, category = get_flower_name(image_bytes=image)
		
		#tensor = get_tensor(image_bytes=image)
		#print(tensor.shape)
	#predicted_flower = 'White Lily'
	return render_template('index.html', flower= flower_name) 
#debug true, application updates at changes, dont need to restart
if __name__ == '__main__':
	app.run(debug=True, port=os.getenv('PORT', 5000))