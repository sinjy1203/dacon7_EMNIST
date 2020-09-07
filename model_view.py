##
from model import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter

##
data_dir = "./datasets"
transform = transforms.Compose([ToTensor()])
dataset = dataset(data_dir=os.path.join(data_dir, "train.csv"), transform=transform)
data = dataset.__getitem__(0)

##
img = data['img'].reshape(1, 1, 28, 28)
letter = data['letter'].reshape(1, 26)

##
net = Net3()
writer = SummaryWriter('./log')
writer.add_graph(net, (img, letter))
writer.close()