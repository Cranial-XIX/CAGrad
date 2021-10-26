import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task: Cross')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='cityscapes', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--seed', default=0, type=int, help='control seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
opt = parser.parse_args()


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 7

        # define encoder decoder layers
        self.encoder_block_t = nn.ModuleList([nn.ModuleList([self.conv_layer([3, filter[0], filter[0]], bottle_neck=True)])])
        self.decoder_block_t = nn.ModuleList([nn.ModuleList([self.conv_layer([filter[0], filter[0], filter[0]], bottle_neck=True)])])

        for j in range(2):
            if j < 1:
                self.encoder_block_t.append(nn.ModuleList([self.conv_layer([3, filter[0], filter[0]], bottle_neck=True)]))
                self.decoder_block_t.append(nn.ModuleList([self.conv_layer([filter[0], filter[0], filter[0]], bottle_neck=True)]))
            for i in range(4):
                if i == 0:
                    self.encoder_block_t[j].append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=True))
                    self.decoder_block_t[j].append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=True))
                else:
                    self.encoder_block_t[j].append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=False))
                    self.decoder_block_t[j].append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=False))

        # define cross-stitch units
        self.cs_unit_encoder = nn.Parameter(data=torch.ones(4, 2))
        self.cs_unit_decoder = nn.Parameter(data=torch.ones(5, 2))

        # define task specific layers
        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], bottle_neck=True, pred_layer=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], bottle_neck=True, pred_layer=True)
        #self.pred_task3 = self.conv_layer([filter[0], 3], bottle_neck=True, pred_layer=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.constant(m.weight, 1)

    def conv_layer(self, channel, bottle_neck, pred_layer=False):
        if bottle_neck:
            if not pred_layer:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel[2]),
                    nn.ReLU(inplace=True),
                )
            else:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
                )

        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[2]),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def forward(self, x):
        encoder_conv_t, decoder_conv_t, encoder_samp_t, decoder_samp_t, indices_t = ([0] * 2 for _ in range(5))
        for i in range(2):
            encoder_conv_t[i], decoder_conv_t[i], encoder_samp_t[i], decoder_samp_t[i], indices_t[i] = ([0] * 5 for _ in range(5))

        # task branch 1
        for i in range(5):
            for j in range(2):
                if i == 0:
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](x)
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(encoder_conv_t[j][i])
                else:
                    encoder_cross_stitch = self.cs_unit_encoder[i - 1][0] * encoder_samp_t[0][i - 1] + \
                                           self.cs_unit_encoder[i - 1][1] * encoder_samp_t[1][i - 1]
                                           #self.cs_unit_encoder[i - 1][2] * encoder_samp_t[2][i - 1]
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](encoder_cross_stitch)
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(encoder_conv_t[j][i])

        for i in range(5):
            for j in range(2):
                if i == 0:
                    decoder_cross_stitch = self.cs_unit_decoder[i][0] * encoder_samp_t[0][-1] + \
                                           self.cs_unit_decoder[i][1] * encoder_samp_t[1][-1]
                                           #self.cs_unit_decoder[i][2] * encoder_samp_t[2][-1]
                    decoder_samp_t[j][i] = self.up_sampling(decoder_cross_stitch, indices_t[j][-i - 1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](decoder_samp_t[j][i])
                else:
                    decoder_cross_stitch = self.cs_unit_decoder[i][0] * decoder_conv_t[0][i - 1] + \
                                           self.cs_unit_decoder[i][1] * decoder_conv_t[1][i - 1]
                                           #self.cs_unit_decoder[i][2] * decoder_conv_t[2][i - 1]
                    decoder_samp_t[j][i] = self.up_sampling(decoder_cross_stitch, indices_t[j][-i - 1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](decoder_samp_t[j][i])

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(decoder_conv_t[0][-1]), dim=1)
        t2_pred = self.pred_task2(decoder_conv_t[1][-1])
        #t3_pred = self.pred_task3(decoder_conv_t[2][-1])
        #t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred], self.logsigma

control_seed(opt.seed)
# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SegNet_CROSS = SegNet().to(device)
optimizer = optim.Adam(SegNet_CROSS.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_CROSS),
                                                         count_parameters(SegNet_CROSS) / 24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
dataset_path = opt.dataroot
if opt.apply_augmentation:
    train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
    print('Applying data augmentation on CityScapes.')
else:
    train_set = CityScapes(root=dataset_path, train=True)
    print('Standard training strategy without data augmentation.')

test_set = CityScapes(root=dataset_path, train=False)

batch_size = 8
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

# Train and evaluate multi-task network
multi_task_trainer(train_loader,
                   test_loader,
                   SegNet_CROSS,
                   device,
                   optimizer,
                   scheduler,
                   opt,
                   200)

