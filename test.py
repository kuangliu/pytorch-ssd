import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from ssd import SSD300
from encoder import DataEncoder
from PIL import Image, ImageDraw


# Load model
net = SSD300()
net.load_state_dict(torch.load('model/net.pth'))
net.eval()

# Load test image
img = Image.open('./img/img2.jpg')
img = img.resize((300,300))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
inputs = transform(img)

# Forward
loc, conf = net(Variable(inputs[None,:,:,:], volatile=True))

# Decode
data_encoder = DataEncoder()
boxes, labels = data_encoder.decode(loc.data.squeeze(0), conf.data.squeeze(0))

draw = ImageDraw.Draw(img)
for box in boxes:
    box *= 300
    draw.rectangle(list(box))

img.show()
