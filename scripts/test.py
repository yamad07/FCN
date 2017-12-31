import torch
import sys
sys.path.append('../')

from modules.data_encoder import DataEncoder
from modules.model import FCN

img_file = '../data/images/0.jpg'
model = FCN()
params = torch.load("../weights/weight_150.pth")
model.load_state_load(params)


transform = transforms.Compose([transforms.Resize((512, 768)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

img = Image.open(img_file)
img = img.to_array()
img = transform(img)

pred = model(img)
encoder = DataEncoder()
pred_img = encoder.decode(pred)

pred_img = pred_img.data.numpy()
pred_img = Image.fromarray(numpy.uint8(pred_img))

pred_img.save("../results/result.jpg")
