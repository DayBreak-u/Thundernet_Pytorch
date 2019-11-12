import torch
from versions.nn.Snet import  get_thundernet
from torch.utils.data import DataLoader
from Dataset.VocGenerator import VocGenerator
from Utils.Engine import train_one_epoch, evaluate
from Utils import Transforms as T
from config import Configs
import  argparse
import  os


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--training_dataset', default= Configs.get("train_txts") , help='Training dataset directory')
parser.add_argument('--val_dataset', default= Configs.get("val_txts") , help='Valing dataset directory')
parser.add_argument('--network', default= Configs.get("Snet_version"), type=int ,  help='49 146 or 535')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--batchsize' , default=16, type=int, help='batchsize ')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
# parser.add_argument('--resume_net_path', default="./save_weights/efficient_rcnn_100.pth", help='resume net path  for retraining')
parser.add_argument('--resume_net_path', default=None, help='resume net path  for retraining')
parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay for SGD')
parser.add_argument('--step_lr', default=[100,150,250], type=float, help='step  for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--num_epochs', default=1000, type=float, help='num_epochs')
parser.add_argument('--save_folder', default='./save_weights/', help='Location to save checkpoint models')

args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = VocGenerator(path= args.training_dataset, type="train", transform=get_transform(train=True))
validation_dataset = VocGenerator(path=args.val_dataset, type="validation", transform=get_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
test_loader = DataLoader(validation_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)



model = get_thundernet()
# put the pieces together inside a FasterRCNN model
if args.resume_net_path is not None:
    print("load weights from {}".format(args.resume_net_path))
    model.load_state_dict(torch.load(args.resume_net_path))

model.eval()
print('Finished loading model!')
print(model)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(params, lr=args.lr,  weight_decay=args.weight_decay)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step_lr, gamma=args.gamma)

# let's train it for 200 epochs
num_epochs = args.num_epochs

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_loader, device=device)
    torch.save(model.state_dict(), "./{}/efficient_rcnn_".format(args.save_folder) + str(epoch) + ".pth")







