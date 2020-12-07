"""
Some algorithm which were used in the PointNet++ such as the normalization
into local centroid, farthest sampling in point cloud, ball query...
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(point_cloud):
    """
    Normalize the point cloud into a sphere with radius 1.

    point_cloud: N points with coordinate (x, y, z)
    """
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    m = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    point_cloud = point_cloud / m
    return point_cloud


def square_distance(src_point_cloud, dst_point_cloud):
    """
    Calculate squared Euclidean distance between each two batch of point clouds.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src_point_cloud.shape
    _, M, _ = dst_point_cloud.shape
    distances = -2 * torch.matmul(src_point_cloud, dst_point_cloud.permute(0, 2, 1))
    distances += torch.sum(src_point_cloud ** 2, -1).view(B, N, 1)
    distances += torch.sum(dst_point_cloud ** 2, -1).view(B, 1, M)
    return distances


def index_points(point_clouds, index):
    """
    给定一批点云points，和一批坐标，取对应坐标的点

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


# def my_index_points(points, index):
#     """
#     自实现批点云中按索引取点
#
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     batch_size, n, c = points.size()
#     _, s = index.size()
#     device = points.device
#     result = torch.zeros(batch_size, s, c, device=device)
#     for i in range(batch_size):
#         result[i] = points[i][index[i]]
#     return result


def farthest_point_sample(point_clouds, sample_num):
    """
    批量最远点采样，返回的是点云中最远点的索引

    Input:
        point_cloud: point cloud data, [B, N, 3]
        sample_num: number of samples
    Return:
        centroids: sampled point cloud index, [B, sample_num]
    """
    device = point_clouds.device
    B, N, C = point_clouds.shape
    centroids = torch.zeros(B, sample_num, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 第一个点随机取
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(sample_num):
        centroids[:, i] = farthest
        centroid = point_clouds[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((point_clouds - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, k, point_clouds, sampled_points):
    """
    批量球查询

    Input:
        radius: local region radius
        k: max sample number in local region
        point_clouds: all points, [B, N, 3]
        sampled_points: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, k]
    """
    device = point_clouds.device
    B, N, C = point_clouds.shape
    _, S, _ = sampled_points.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    squared_dist = square_distance(sampled_points, point_clouds)
    group_idx[squared_dist > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :k]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, k])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(num_fps, radius, k, cloud_points, cloud_points_features, return_fps=False):
    """
    Input:
        num_fps: 每个点云降采样多少个点
        radius: 球查询的半径
        k: 每个邻域内多少个点
        cloud_points: input points position data, [B, N, 3]
        cloud_points_features: input points data, [B, N, D]
    Return:
        fps_point_clouds: sampled points position data, [B, num_fps, 3]
        new_point_clouds_features: sampled points data, [B, num_fps, k, 3+D]
    """
    B, N, C = cloud_points.shape
    S = num_fps

    # FPS sampling
    fps_idx = farthest_point_sample(cloud_points, num_fps)  # [B, S]
    torch.cuda.empty_cache()
    fps_point_clouds = index_points(cloud_points, fps_idx)  # [B, S, C]
    torch.cuda.empty_cache()

    # Ball Query
    ball_query_index = query_ball_point(radius, k, cloud_points, fps_point_clouds)  # [B, S, k]
    torch.cuda.empty_cache()
    ball_query_xyz = index_points(cloud_points, ball_query_index)  # [B, S, k, C]
    torch.cuda.empty_cache()

    # normalize into local coordinate
    norm_ball_query_xyz = ball_query_xyz - fps_point_clouds.view(B, S, 1, C)  # [B, S, k, C]
    torch.cuda.empty_cache()

    if cloud_points_features is not None:
        ball_query_features = index_points(cloud_points_features, ball_query_index)
        new_point_clouds_features = torch.cat([norm_ball_query_xyz, ball_query_features], dim=-1)  # [B, S, k, C+D]
    else:
        new_point_clouds_features = norm_ball_query_xyz  # [B, S, k, C]
    if return_fps:
        return fps_point_clouds, new_point_clouds_features, ball_query_xyz, fps_idx
    else:
        return fps_point_clouds, new_point_clouds_features  # [B, S, C], [B, S, k, C+D]


def sample_and_group_all(point_clouds, point_clouds_features):
    """
    Only FPS sample one point and do the ball query.
    The FPS result is the original point (0, 0, 0), and the ball query contains
    all point in the point cloud.

    Input:
        point_clouds: input points position data, [B, N, 3]
        point_clouds_features: input points data, [B, N, D]
    Return:
        fps_point_clouds: sampled points position data, [B, 1, 3]
        new_point_clouds_features: sampled points data, [B, 1, N, 3+D]
    """
    device = point_clouds.device
    B, N, C = point_clouds.shape
    fps_point_clouds = torch.zeros(B, 1, C).to(device)
    norm_ball_query_xyz = point_clouds.view(B, 1, N, C)
    if point_clouds_features is not None:
        new_point_clouds_features = torch.cat([norm_ball_query_xyz, point_clouds_features.view(B, 1, N, -1)], dim=-1)
    else:
        new_point_clouds_features = norm_ball_query_xyz
    return fps_point_clouds, new_point_clouds_features  # [B, 1, 3], [B, 1, N, 3(+D)]


class SA(nn.Module):
    """
    Set Abstraction Layer, which samples the points with FPS algorithm, grouping point cloud into
    many local neighborhood regions, and then extract features by PointNet.

    num_fps:    number of unsampling points with FPS
    k:          the limitation of number of points in the local ball neighborhood region
    radius:     the radius of the ball neighborhood region
    in_channel: the channel of input features, may be 3, 3+D, or D
    mlp:        the output channels of the shared mlp of PointNet
    group_all:  whether in the hierarchical architecture of the PointNet++
    """

    def __init__(self, num_fps, radius, k, in_channel, mlp, group_all=False):
        super().__init__()
        self.num_fps = num_fps
        self.radius = radius
        self.k = k
        self.group_all = group_all

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, bpc, bpc_features):
        bpc = bpc.permute(0, 2, 1)
        if bpc_features is not None:
            bpc_features = bpc_features.permute(0, 2, 1)

        if self.group_all:
            fps_point_clouds, new_point_clouds_features = sample_and_group_all(bpc, bpc_features)
        else:
            fps_point_clouds, new_point_clouds_features = sample_and_group(self.num_fps, self.radius,
                                                                           self.k, bpc, bpc_features)

        x = new_point_clouds_features.permute(0, 3, 1, 2)  # [B, C'=C(+D), S, k]

        # PointNet, size was transformed to [B, C'', S, k], and then
        # was transformed to [B, C'', S]
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = torch.max(x, dim=-1)[0]  # [B, C'', S]

        fps_point_clouds = fps_point_clouds.permute(0, 2, 1)  # [B, C, S]

        return fps_point_clouds, x


class MSGSA(nn.Module):
    def __init__(self, num_fps, radius_list, ks, in_channel, mlp_list):
        super().__init__()
        self.num_fps = num_fps
        self.radius_list = radius_list
        self.ks = ks

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i, mlp in enumerate(mlp_list):
            last_channel = in_channel
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, bpc, bpc_features):
        b, c, n = bpc.size()
        s = self.num_fps
        bpc = bpc.permute(0, 2, 1)
        if bpc_features is not None:
            _, d, _ = bpc_features.size()
            bpc_features = bpc_features.permute(0, 2, 1)

        # 采样是一致的
        fps_points = index_points(bpc, farthest_point_sample(bpc, s))  # [B, S, C]
        new_features_list = []

        # MSG
        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            k = self.ks[i]
            conv_block = self.conv_blocks[i]
            bn_block = self.bn_blocks[i]

            # 划分邻域，局部归一化
            group_idx = query_ball_point(radius, k, bpc, fps_points)
            batch_grouped_points = index_points(bpc, group_idx)
            batch_grouped_points -= fps_points.view(b, s, 1, c)  # [B, S, k, C]

            if bpc_features is not None:
                batch_grouped_features = index_points(bpc_features, group_idx)
                batch_grouped_features = torch.cat([batch_grouped_features, batch_grouped_points], dim=-1)
            else:
                batch_grouped_features = batch_grouped_points

            x = batch_grouped_features.permute(0, 3, 1, 2)  # [B, C', S, k]

            # shared mlp and symmetric function in PointNet, [B, C'', S, k]
            for j, (conv, bn) in enumerate(zip(conv_block, bn_block)):
                x = F.relu(bn(conv(x)))
            x = torch.max(x, dim=-1)[0]

            new_features_list.append(x)

        new_features = torch.cat(new_features_list, dim=1)  # [B, C'', S]

        fps_points = fps_points.permute(0, 2, 1)  # [B, C, S]

        return fps_points, new_features


class FP(nn.Module):
    """
    Feature Propagation Layer, including interpolation and unit PointNet.
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N], l-1 layer
            xyz2: sampled input points position data, [B, C, S], l layer
            points1: input points data, [B, D, N], l-1 layer
            points2: input points data, [B, D', S], l layer
        Return:
            new_points: upsampled points data, [B, D'', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # interpolation from l layer to l-1 layer
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # [B, N, D]

        # skip links
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        # unit PointNet
        for i, conv in enumerate(self.convs):
            bn = self.bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


if __name__ == '__main__':
    x = torch.rand((32, 3, 1024))
    features = torch.rand((32, 3, 1024))

    """
    Testing for SA
    """
    # without extra features
    sa = SA(512, 0.2, 32, 3, [64, 64, 128], False)
    new_x, new_features = sa(x, None)
    print(new_x.size(), new_features.size())
    # adding normal features
    sa = SA(512, 0.2, 32, 6, [64, 64, 128], False)
    new_x, new_features = sa(x, features)
    print(new_x.size(), new_features.size())
    print("=========================================================")

    """
    Testing for MSGSA
    """
    # without extra features
    msg_sa = MSGSA(512, [0.1, 0.2, 0.4], [16, 32, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
    fps_points, new_features = msg_sa(x, None)
    print(fps_points.size(), new_features.size())
    # adding normal features
    msg_sa = MSGSA(512, [0.1, 0.2, 0.4], [16, 32, 128], 6, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
    fps_points, new_features = msg_sa(x, features)
    print(fps_points.size(), new_features.size())
    print("=========================================================")

    """
    Testing for FP
    """
    x0 = torch.rand((16, 3, 1024))
    x0_features = torch.rand((16, 3, 1024))
    x1 = torch.rand((16, 3, 512))
    x1_features = torch.rand((16, 64, 512))
    # without extra feature for layer l-1
    fp = FP(64, [256, 256])
    result = fp(x0, x1, None, x1_features)
    print(result.size())
    # add extra feature for layer l-1
    fp = FP(3 + 64, [256, 256])
    result = fp(x0, x1, x0_features, x1_features)
    print(result.size())
