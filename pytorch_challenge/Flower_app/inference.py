import json
from commons import get_tensor, load_model

with open('cat_to_name.json') as f:
	cat_to_name=json.load(f)

model = load_model('classifier.pth')
class_to_idx = model.class_to_idx
#print(class_to_idx)
idx_to_class = {val: key for key, val in class_to_idx.items()}

def get_flower_name(image_bytes):
	tensor = get_tensor(image_bytes)
	outputs = model.forward(tensor)
	#print(outputs)
	_, prediction = outputs.max(1)
	category = prediction.item() 
	flower_name = cat_to_name[idx_to_class[category]]
	return flower_name, category


