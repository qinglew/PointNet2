"""
SSG分类网络、SSG分割网络
MSG分类网络、MSG分割网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import SA, MSGSA, FP


class SSGClassification(nn.Module):
    """
    SSG PointNet++ for classification task. It's architecture is:
    SA(512, 0,2, [64,64,128]) -> SA(128, 0.4, [128,128,256]) -> SA([256,512,1024]) ->
    FC(512, 0.5) -> FC(256, 0.5) -> FC(K)
    """

    def __init__(self, categories, normal=False):
        super().__init__()
        self.categories = categories
        self.normal = normal

        in_channel = 6 if normal else 3

        self.sa1 = SA(512, 0.2, 50, in_channel, [64, 64, 128])
        self.sa2 = SA(128, 0.4, 50, in_channel=128+3, mlp=[128, 128, 256])
        self.sa3 = SA(None, None, None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, categories)

    def forward(self, x):
        if self.normal:
            normals = x[:, 3:, :]
            x = x[:, :3, :]
        else:
            normals = None

        fps_points1, new_features1 = self.sa1(x, normals)
        fps_points2, new_features2 = self.sa2(fps_points1, new_features1)
        fps_points3, new_features3 = self.sa3(fps_points2, new_features2)

        x = new_features3.view(-1, 1024)

        x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
        x = F.relu(self.bn2(self.dropout2(self.fc2(x))))

        x = F.log_softmax(self.fc3(x), dim=-1)

        return x


class MSGClassification(nn.Module):
    """
    MSG PointNet++ for classification task, whose architecture are blew:
    SA(512, [0.1, 0.2, 0.4], [[32,32,64],[64,64,218],[64,96,128]] ->
    SA(128, [0.2, 0.4, 0.8], [[64,64,128],[128,128,256],[128,128,256]] ->
    SA([256,512,1024]) -> FC(512,0.5) -> FC(256,0.5) -> FC(K)
    """

    def __init__(self, categories, normal=False):
        super().__init__()
        self.categories = categories
        self.normal = normal
        in_channel = 6 if self.normal else 3

        self.msg_sa1 = MSGSA(512, [0.1, 0.2, 0.4], [50, 50, 50], in_channel=in_channel,
                             mlp_list=[[32, 32, 64], [64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.msg_sa2 = MSGSA(128, [0.2, 0,4, 0.8], [50, 50, 50], in_channel=323,
                             mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = SA(None, None, None, in_channel=643, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, categories)

    def forward(self, x):
        if self.normal:
            # split into (x, y, z) and normal vector(n1, n2, n3)
            normals = x[:, 3:, :]
            x = x[:, :3, :]
        else:
            normals = None

        # shared mlp
        new_xyz1, new_features1 = self.msg_sa1(x, normals)
        new_xyz2, new_features2 = self.msg_sa2(new_xyz1, new_features1)
        new_xyz3, new_features3 = self.sa3(new_xyz2, new_features2)

        x_ = new_features3.view(-1, 1024)

        # fully connected layers and softmax layer
        x_ = F.relu(self.bn1(self.dropout1(self.fc1(x_))))
        x_ = F.relu(self.bn2(self.dropout2(self.fc2(x_))))
        x_ = F.log_softmax(self.fc3(x_), dim=-1)

        return x_


class SSGSemanticSegmentation(nn.Module):
    """
    SSG PointNet++ for scene semantic segmentation. The architecture is blow:
    SA(1024, 0.1, [32,32,64]) -> SA(256, 0.2, [64,64,128]) -> SA(64, 0.4, [128, 128, 256]) ->
    SA(16, 0.8, [256, 256, 512]) -> FP(256, 256) -> FP(256, 256) -> FP(128, 128, 128, 128, k)
    """

    def __init__(self, categories, normal=False):
        super().__init__()
        self.categories = categories
        self.normal = normal
        in_channel = 6 if self.normal else 3

        self.sa1 = SA(1024, 0.1, 50, in_channel, mlp=[32, 32, 64])
        self.sa2 = SA(256, 0.2, 50, in_channel=64+3, mlp=[64, 64, 128])
        self.sa3 = SA(64, 0.4, 50, in_channel=128+3, mlp=[128, 128, 256])
        self.sa4 = SA(16, 0.8, 50, in_channel=256+3, mlp=[256, 256, 512])

        self.fp1 = FP(512+256, [256, 256])
        self.fp2 = FP(256+128, [256, 256])
        self.fp3 = FP(256+64, [256, 128])

        if self.normal is not None:
            self.fp4 = FP(128+3, [128, 128, 128])
        else:
            self.fp4 = FP(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(128, categories, 1)

    def forward(self, x):
        if self.normal:
            normals = x[:, 3:, :]
            x_ = x[:, :3, :]
        else:
            normals = None
            x_ = x

        # extract features
        new_xyz1, new_features1 = self.sa1(x_, normals)
        new_xyz2, new_features2 = self.sa2(new_xyz1, new_features1)
        new_xyz3, new_features3 = self.sa3(new_xyz2, new_features2)
        new_xyz4, new_features4 = self.sa4(new_xyz3, new_features3)

        # interpolation
        l3_features = self.fp1(new_xyz3, new_xyz4, new_features3, new_features4)
        l2_features = self.fp2(new_xyz2, new_xyz3, new_features2, l3_features)
        l1_features = self.fp3(new_xyz1, new_xyz2, new_features1, l2_features)
        ret = self.fp4(x, new_xyz1, normals, l1_features)

        # two fully connected layers
        ret = F.relu(self.bn1(self.drop1(self.conv1(ret))))
        ret = F.log_softmax(self.conv2(ret), dim=1)

        ret = ret.permute(0, 2, 1)

        return ret


class SSGPartSegmentation(nn.Module):
    """
        SSG PointNet++ for part segmentation. The architecture is blow:
        SA(512, 0.2, [64,64,128]) -> SA(128, 0.4, [128,128,256]) -> SA([256, 512, 1024]) ->
        FP(256, 256) -> FP(256, 128) -> FP(128, 128, 128, 128, k)
    """

    def __init__(self, categories, normal=False):
        super().__init__()
        self.categories = categories
        self.normal = normal
        in_channel = 6 if self.normal else 3

        self.sa1 = SA(512, 0.2, 50, in_channel, mlp=[64, 64, 128])
        self.sa2 = SA(128, 0.4, 50, in_channel=128+3, mlp=[128, 128, 256])
        self.sa3 = SA(None, None, None, in_channel=256+3, mlp=[256, 512, 1024])

        self.fp1 = FP(1024+256, [256, 256])
        self.fp2 = FP(256+128, [256, 128])

        if self.normal is not None:
            self.fp3 = FP(128+3, [128, 128, 128])
        else:
            self.fp3 = FP(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(128, categories, 1)

    def forward(self, x):
        if self.normal:
            normals = x[:, 3:, :]
            x_ = x[:, :3, :]
        else:
            normals = None
            x_ = x

        # extract features in hierarchical architecture of PointNet++
        l1_xyz, l1_features = x_, normals
        l2_xyz, l2_features = self.sa1(l1_xyz, normals)
        l3_xyz, l3_features = self.sa2(l2_xyz, l2_features)
        l4_xyz, l4_features = self.sa3(l3_xyz, l3_features)

        # features interpolation
        l3_new_features = self.fp1(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_new_features = self.fp2(l2_xyz, l3_xyz, l2_features, l3_new_features)
        l1_new_features = self.fp3(l1_xyz, l2_xyz, l1_features, l2_new_features)

        # two fully connected layers
        ret = F.relu(self.bn1(self.drop1(self.conv1(l1_new_features))))
        ret = F.log_softmax(self.conv2(ret), dim=1)

        ret = ret.permute(0, 2, 1)

        return ret


if __name__ == '__main__':
    """
    Testing for SSG classification network
    """
    net = SSGClassification(10)
    point_cloud = torch.rand((10, 3, 1024))
    result = net(point_cloud)
    print(result.size())
