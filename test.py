import torch
import torchvision
import torch.nn.functional as F
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
img = Image.open('./image/img1.jpg')
img1 = img.resize((300,300))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
img1 = transform(img1)

# Forward
loc, conf = net(Variable(img1[None,:,:,:], volatile=True))

# Decode
data_encoder = DataEncoder()
boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)

draw = ImageDraw.Draw(img)
for box in boxes:
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), outline='red')
img.show()
