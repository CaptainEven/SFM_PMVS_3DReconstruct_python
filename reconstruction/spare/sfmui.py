# encoding=utf-8

import cv2
import numpy as np
import os
import math
from mayavi import mlab
from tqdm import tqdm
from scipy.optimize import least_squares
from reconstruction import rec_config
import glob
import shutil
from scipy.sparse import lil_matrix
from calibration import calibration

cali_img_dir = calibration.cali_img_dir


def extract_feathers(image_names):
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.AKAZE_create()

    keypoints_for_all = []
    descriptors_for_all = []
    colors_for_all = []

    for image_name in tqdm(image_names):
        image = cv2.imread(image_name)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ---------- SIFT extractor
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        # ----------

        # 添加该图片的关键点信息
        keypoints_for_all.append(keypoints)
        descriptors_for_all.append(descriptors)

        # 获取关键点颜色信息
        colors = np.zeros((len(keypoints), 3))
        for i, keypoint in enumerate(keypoints):
            p = keypoint.pt
            colors[i] = image[int(p[1])][int(p[0])]
        colors_for_all.append(colors)

    return np.array(keypoints_for_all), np.array(descriptors_for_all), np.array(colors_for_all)


# 特征点匹配
def match_feathers(kp1, kp2, des1, des2):
    # 用FLANNE算法对两张照片进行关键点匹配，返回这两张照片的匹配对
    # FLANN参数设计
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 核密度树数量
    # search_params = dict(checks=50)  # 遍历次数

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)  # NORM_HAMMING, NORM_HAMMING2,  NORM_L2
    matches = bf.knnMatch(des1, des2, k=2)

    # 设置两距离比值小于0.7时为可用匹配  （Lowe's ratio test）
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        # RANSAC随机采样一致
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)

        # 将mask变成一维数组
        mask = mask.ravel().tolist()

        i = 0
        j = 0
        while (j < len(mask)):
            if mask[j] == 0:
                good.remove(good[i])
                j += 1
            else:
                i += 1
                j += 1

    else:
        print("not have enough matches!")

    return np.array(good)


# 对相邻图片特征匹配
def match_all_feather(keypoints_for_all, descriptors_for_all):
    # 只需相邻图片进行匹配 1-2 2-3 ...N-1 - N 一共N-1对匹配
    matches_for_all = []
    for i in tqdm(range(len(descriptors_for_all) - 1)):
        matches = match_feathers(keypoints_for_all[i], keypoints_for_all[i + 1], descriptors_for_all[i],
                                 descriptors_for_all[i + 1])
        matches_for_all.append(matches)

    return np.array(matches_for_all)


# ============Part 2==========
# 计算本质矩阵和外参
def find_transform(K, p1, p2):
    # 根据匹配点求本征矩阵，使用RANSAC进一步消除失配点
    # 基于五点算法 mask用来标记RANSAC计算得来的内点 只有内点才被用于后续步骤
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)

    # 分解本征矩阵E, 得到位姿R, T
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, K, mask)

    return R, T, mask


# 三维重建
def reconstruction(K, R1, T1, R2, T2, p1, p2):
    # 设置两个相机的投影矩阵[R T]，且转为float型数据 triangulatePoints函数只支持float
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))

    proj1[0:3, 0:3] = np.float32(R1)  # 将R1赋值给proj1的对应位置（前三行三列）
    proj1[:, 3] = np.float32(T1.T)  # 将T1的转置赋值给proj1的对应位置（第四列）
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)

    fk = np.float32(K)

    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)

    # p1 p2 原本Nx2 转置为2*N
    pts4d = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)

    structure = []
    for i in range(pts4d.shape[1]):  # 列数表示计算出来空间点的个数 将三角化的结果进行处理得到“正常”的点坐标
        col = pts4d[:, i]
        col = col / col[3]
        structure.append([col[0], col[1], col[2]])

    return np.array(structure), proj2


def reconFrom2Views(K1, K2, R1, T1, R2, T2, p1, p2):
    """
    :param K1:
    :param K2:
    :param R1:
    :param T1:
    :param R2:
    :param T2:
    :param p1:
    :param p2:
    :return:
    """
    # 设置两个相机的投影矩阵[R T]，且转为float型数据 triangulatePoints函数只支持float
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))

    proj1[0:3, 0:3] = np.float32(R1)  # 将R1赋值给proj1的对应位置（前三行三列）
    proj1[:, 3] = np.float32(T1.T)  # 将T1的转置赋值给proj1的对应位置（第四列）
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)

    fk1 = np.float32(K1)
    fk2 = np.float32(K2)

    proj1 = np.dot(fk1, proj1)
    proj2 = np.dot(fk2, proj2)

    # p1 p2 原本Nx2 转置为2*N
    pts4d = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)

    pts3d = []
    for i in range(len(pts4d[0])):  # 列数表示计算出来空间点的个数 将三角化的结果进行处理得到“正常”的点坐标
        col = pts4d[:, i]
        col /= col[3]
        pts3d.append([col[0], col[1], col[2]])

    return np.array(pts3d), proj2


# 获取匹配点坐标
def get_matched_points(p1, p2, matches):
    out_p1 = np.asarray([p1[m.queryIdx].pt for m in matches])  # queryIndex 查询图像中描述符的索引
    out_p2 = np.asarray([p2[m.trainIdx].pt for m in matches])

    return out_p1, out_p2


# 获取匹配点颜色信息
def get_matched_colors(c1, c2, matches):
    color1 = np.asarray([c1[m.queryIdx] for m in matches])
    color2 = np.asarray([c2[m.trainIdx] for m in matches])

    return color1, color2


# 获取内点的坐标
def maskout_point(p, mask):
    p_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p_copy.append(p[i])

    return np.array(p_copy)


# 获取内点颜色信息
def maskout_color(c, mask):
    c_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            c_copy.append(c[i])

    return np.array(c_copy)


# ============Part 3==========
# 初始化点云
def init_structure(K, keypoints_for_all, colors_for_all, matches_for_all):
    """
    :param K:
    :param keypoints_for_all:
    :param colors_for_all:
    :param matches_for_all:
    :return:
    """
    # 找到前两幅图的关键点坐标和相应的颜色信息: matches_for_all[0] 为第一幅图和第二幅图之间的匹配对
    p1, p2 = get_matched_points(keypoints_for_all[0], keypoints_for_all[1], matches_for_all[0])
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])

    # 计算外参
    if find_transform(K, p1, p2):
        R, T, mask = find_transform(K, p1, p2)
    else:
        R, T, mask = np.array([]), np.array([]), np.array([])

    # 对前两幅图进行三维重建
    p1 = maskout_point(p1, mask)
    p2 = maskout_point(p2, mask)
    colors = maskout_color(c1, mask)

    R0 = np.eye(3, 3, dtype=np.float)  # 设置第一个相机的变换矩阵，作为剩下摄像机矩阵变换的基准
    T0 = np.zeros((3, 1), dtype=np.float)
    proj0 = np.zeros((3, 4), dtype=np.float)  # 第一个相机的投影矩阵
    proj0[0:3, 0:3] = np.identity(3, dtype=np.float)  # 将R1赋值给proj1的对应位置（前三行三列）
    proj0[:, 3] = np.ones(3, dtype=np.float)
    fk = np.float32(K)
    proj0 = np.dot(fk, proj0)
    projections = [proj0]

    structure, proj = reconstruction(K, R0, T0, R, T, p1, p2)
    projections.append(proj)  # 加入第二幅图片的投影矩阵

    rotations = [R0, R]
    motions = [T0, T]

    correspond_struct_idx = []  # correspond_struct_idx[i][j]表示第i幅图的第j个特征点对应的空间点在点云中的索引值（空间点云中第几个点？）
    for kp in keypoints_for_all:
        correspond_struct_idx.append(np.ones(len(kp)) * -1)  # 初始化correspond_struct_idx的大小 图片数*相应图片特征点数目 各元素初始值为-1

    # 填写前两幅图特征点的索引
    idx = 0  # 初始化空间点云索引值
    for i, match in enumerate(matches_for_all[0]):
        if mask[i] == 0:
            continue

        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[1][int(match.trainIdx)] = idx
        idx += 1

    return structure, correspond_struct_idx, colors, rotations, motions, projections


# 获取空间点和像素点用于solvePnPRansac
def get_objpts_and_imgpts(matches, struct_indices, structure, keypoints):
    # matches是第i幅图与第i+1幅图之间的匹配，struct_indices是第i幅图特征点对应的空间点索引，structure为已有点云，keypoints为第i+1幅图的关键点
    obj_points = []
    img_points = []

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]

        if struct_idx < 0:
            continue

        obj_points.append(structure[int(struct_idx)])
        img_points.append(keypoints[int(train_idx)].pt)

    return np.array(obj_points), np.array(img_points)


# 点云融合
def fuse_structure(matches,
                   struct_indict1, struct_indict2,
                   structure, new_structure,
                   colors, new_colors):
    """
    :param matches:
    :param struct_indict1:
    :param struct_indict2:
    :param structure:
    :param new_structure:
    :param colors:
    :param new_colors:
    :return:
    """
    # matches：图i与图i+1之间的匹配
    # struct_indict1：图i的特征点在点云中的索引，struct_indict2：图i+1的特征点在点云中的索引
    # structure:原有的点云 new_structure:新生成的三维点
    # colors: 原有点云各点的颜色，new_colors：新生成点云各点的颜色
    for i, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indict1[query_idx]

        # 如果新生成的三维点已存在，则告诉图i+1是哪个空间点
        if struct_idx >= 0:
            struct_indict2[train_idx] = struct_idx
            continue

        # 如果新生成的三维点不存在于原有点云中，则加入新点
        structure = np.append(structure, [new_structure[i]], axis=0)
        colors = np.append(colors, [new_colors[i]], axis=0)
        struct_indict1[query_idx] = struct_indict2[train_idx] = len(structure) - 1

    return structure, colors, struct_indict1, struct_indict2


# ============Part 4==========
# BA优化部分

# 根据重映射误差优化三维点
# 1 删除误差大的点
def get_3dpoints_v1(obj_p, img_p, R, T, K):
    p, J = cv2.projectPoints(obj_p.reshape(1, 1, 3), R, T, K, np.array([]))
    p = p.reshape(2)
    e = img_p - p  # 计算误差
    if e[0] > 1.0 or e[1] > 1.0:  # 误差太大则不要这个点
        return None

    return obj_p  # 只返回误差满足要求的点


# 2 对整个点云进行BA优化
def reproj_err(rotations, motions, K,
               struct_index, key_points_for_all, structure,
               distort_coefs=[]):
    """
    :param rotations:
    :param motions:
    :param K:
    :param struct_index:
    :param key_points_for_all:
    :param structure:
    :param distort_coefs:
    :return:
    """
    if distort_coefs == []:
        print('\nIgnore distortion coefficients.')
    else:
        print('\nTaking distortion into consideration.')

    # 这里需要矩阵
    errors = 0.0
    num = 0

    # 这里需要的是旋转向量
    for i in tqdm(range(len(struct_index))):
        # 获取每张图片特征点、对应的点云索引、旋转向量、平移矩阵
        pt3d_inds = struct_index[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]

        for j in range(len(pt3d_inds)):
            pt3d_idx = int(pt3d_inds[j])
            if pt3d_idx < 0:
                continue

            num += 1

            obj_pt = structure[pt3d_idx].reshape(1, 1, 3)
            img_pt = key_points[j].pt
            r_vec, _ = cv2.Rodrigues(r)

            # ---------- 3D ——>(重投影) 2D: 这里没有考虑畸变的影响
            if distort_coefs == []:
                pt2d, jacob = cv2.projectPoints(obj_pt, r_vec, t, K, np.array([]))
            else:
                pt2d, jacob = cv2.projectPoints(obj_pt, r_vec, t, K, distort_coefs)
            pt2d = pt2d.reshape(2)
            e = img_pt - pt2d  # 计算误差
            # dist = math.sqrt((img_pt[0] - pt2d[0]) * (img_pt[0] - pt2d[0])
            #                + (img_pt[1] - pt2d[1]) * (img_pt[1] - pt2d[1]))
            error = (e[0] + e[1]) * 0.5  # x,y 误差均值
            errors += error

    avg_error = errors / num
    print("点云数目：{}, 平均误差：{} pixel".format(len(structure), abs(avg_error)))


def delete_error_point(rotations, motions, K, struct_index, key_points_for_all, structure):
    # BA优化
    for i in tqdm(range(len(struct_index))):
        # 获取每张图片特征点、对应的点云索引、旋转向量、平移矩阵
        point3d_idxs = struct_index[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]

        # 对该图片包含的空间点进行优化
        for j in range(len(point3d_idxs)):
            pt3d_idx = int(point3d_idxs[j])
            if pt3d_idx < 0:
                continue
            new_point = get_3dpoints_v1(structure[pt3d_idx], key_points[j].pt, r, t, K)
            structure[pt3d_idx] = new_point

    return structure


# 计算残差: 是由观测值减预测值定义的
# Function which computes the vector of residuals
def func(all_params,
         n_cams,
         n_pts,
         inds_2d_to_3d,
         kts_for_all,
         K,
         distort_coefs=[]):
    """
    :param all_params:
    :param n_cams:
    :param n_pts:
    :param inds_2d_to_3d:
    :param kts_for_all:
    :param K: camera intrinsics
    :param distort_coefs:
    :return:
    """
    # param: rotations, motions, 以及 points的扁平列表[rotation*3, motion*3, point*3]
    """Compute residuals.

    `all_params` contains camera parameters and 3-D coordinates.
    """

    errs = []
    rotations = all_params[:n_cams * 3].reshape((n_cams, 3))  # 旋转向量
    motions = all_params[n_cams * 3: n_cams * 6].reshape((n_cams, 3))  # 平移向量
    pts3d = all_params[n_cams * 6:n_cams * 6 + n_pts * 3].reshape((n_pts, 3))  # 3D点

    # 相机内参
    K_ = all_params[n_cams * 6 + n_pts * 3:n_cams * 6 + n_pts * 3 + 9].reshape((3, 3))
    K = K_

    # 畸变系数
    distort_coefs_ = all_params[n_cams * 6 + n_pts * 3 + 9:n_cams * 6 + n_pts * 3 + 14].reshape((1, 5))
    distort_coefs = distort_coefs_

    pt3d_cnt = 0
    for view_i in range(len(inds_2d_to_3d)):
        # 获取每张图片特征点、对应的点云索引、旋转向量、平移矩阵
        pt3d_inds = inds_2d_to_3d[view_i]
        key_points = kts_for_all[view_i]
        r = rotations[view_i]
        t = motions[view_i]

        for j in range(len(pt3d_inds)):
            pt3d_idx = int(pt3d_inds[j])

            if pt3d_idx < 0:
                continue

            pt3d_cnt += 1

            obj_p = pts3d[pt3d_idx].reshape(1, 1, 3)
            img_p = key_points[j].pt

            ## ---------- 3D ——> 2D
            if distort_coefs == []:  # 这里没有考虑畸变因素
                est_p, J = cv2.projectPoints(obj_p, r, t, K_, np.array([]))
            else:  # 考虑畸变的影响
                est_p, J = cv2.projectPoints(obj_p, r, t, K_, np.array(distort_coefs_))

            est_p = est_p.reshape(2)

            # 观测 - 预测
            err = img_p - est_p

            errs.append(err[0])
            errs.append(err[1])

    # print("Total {:d} 3D points' re-projection error computed.".format(pt3d_cnt))
    return np.array(errs)


'''
lil_matrix:基于行连接存储的稀疏矩阵(Row-based linked list sparse matrix)
优点：

快速按行切片
高效地添加、删除、查找元素
缺点:

按列切片很慢（建议CSC）
算术运算LIL+LIL很慢（考虑CSR或CSC）
'''


def bundle_adjustment_sparsity(n_cameras, n_points, camera_inds, pt3d_inds):
    """
    :param n_cameras:  number of views
    :param n_points:   number of 3D points
    :param camera_inds:    number: 2D-3D mapping pairs (number of 2D feature points: n_observations)
    :param pt3d_inds:  number: 2D-3D mapping pairs
    :return:
    """
    # TODO: 雅可比稀疏矩阵

    m = camera_inds.size * 2  # rows number: n_observations(mappings) * 2(x, y)
    n = n_cameras * 6 + n_points * 3 + 9 + 5  # cols number:
    A = lil_matrix((m, n), dtype=int)

    obs_inds = np.arange(camera_inds.size)
    for i in range(6):  # camera pose: rotations and translations
        A[2 * obs_inds, camera_inds * 6 + i] = 1
        A[2 * obs_inds + 1, camera_inds * 6 + i] = 1

    for i in range(3):  # 3D points
        A[2 * obs_inds, n_cameras * 6 + pt3d_inds * 3 + i] = 1
        A[2 * obs_inds + 1, n_cameras * 6 + pt3d_inds * 3 + i] = 1

    for i in range(9):  # camera intrinsics
        A[2 * obs_inds, n_cameras * 6 + n_points * 3 + i] = 1
        A[2 * obs_inds + 1, n_cameras * 6 + n_points * 3 + i] = 1

    for i in range(5):  # camera distortion coefficients
        A[2 * obs_inds, n_cameras * 6 + n_points * 3 + 9 + i] = 1
        A[2 * obs_inds + 1, n_cameras * 6 + n_points * 3 + 9 + i] = 1

    return A


def read_data(inds_mapping_2d_3d_per_view):
    """
    :param inds_mapping_2d_3d_per_view:
    :return:
    """
    camera_inds = np.array([])
    pt3d_inds = np.array([])

    valid_mapping_cnt = 0
    for view_i in range(len(inds_mapping_2d_3d_per_view)):
        for pt2d_j in range(len(inds_mapping_2d_3d_per_view[view_i])):
            if inds_mapping_2d_3d_per_view[view_i][pt2d_j] > -1:  # for validate 2D-3D points
                valid_mapping_cnt += 1

                camera_inds = np.append(camera_inds, view_i)
                pt3d_inds = np.append(pt3d_inds, inds_mapping_2d_3d_per_view[view_i][pt2d_j])

    print('Total {:d} valid 2D-3D mappings.'.format(valid_mapping_cnt))
    return camera_inds, pt3d_inds


def BA(pts3d,
       inds_2d_to_3d,
       mot_vects, rotate_mats,
       kpts_for_all,
       K,
       distort_coefs=[]):
    """
    :param pts3d:
    :param inds_2d_to_3d:
    :param mot_vects:
    :param rotate_mats:
    :param kpts_for_all:
    :param K:
    :param distort_coefs:
    :return:
    """
    if distort_coefs == []:
        print('\nIgnore distortion coefficients.')
    else:
        print('\nTaking distortion into consideration.')

    camera_inds, pt3d_inds = read_data(inds_2d_to_3d)
    n_cams = len(mot_vects)
    n_pts = len(pts3d)

    # 这里需要的是向量
    rot_vecs = []
    for i in range(len(rotate_mats)):
        R, _ = cv2.Rodrigues(rotate_mats[i])  # 旋转矩阵转向量
        rot_vecs.append(R)
    rot_vecs = np.array(rot_vecs)

    # ----------
    # flatten()分配了新的内存, 但ravel()返回的是一个1维数组的视图
    # all_params = np.hstack((rot_vecs.ravel(), mot_vects.ravel(), pts3d.ravel()))  # all params need to be optimized
    all_params = np.hstack((rot_vecs.ravel(), mot_vects.ravel(), pts3d.ravel(), K.ravel(), distort_coefs.ravel()))

    residual_errs = func(all_params, n_cams, n_pts, inds_2d_to_3d, kpts_for_all, K, distort_coefs)
    A = bundle_adjustment_sparsity(n_cams, n_pts, camera_inds, pt3d_inds)
    res = least_squares(func,
                        all_params,
                        jac_sparsity=A,  # A
                        verbose=2,
                        x_scale='jac', method='trf', loss='linear',
                        args=(n_cams, n_pts, inds_2d_to_3d, kpts_for_all, K, distort_coefs))

    # ----------

    # ---------- output
    new_x = res.x
    pts3d = np.array(new_x[n_cams * 6:n_cams * 6 + n_pts * 3]).reshape((-1, 3))
    K_new = np.array(new_x[n_cams * 6 + n_pts * 3:n_cams * 6 + n_pts * 3 + 9]).reshape((3, 3))
    dist_coefs_new = np.array(new_x[n_cams * 6 + n_pts * 3 + 9:n_cams * 6 + n_pts * 3 + 14]).reshape((1, 5))
    poses_new = new_x[: n_cams * 6]

    print('K refined:\n', K_new)
    print('Distortion coefficients refined:\n', dist_coefs_new)

    # TODO: 这里可以输出现有误差
    return pts3d, poses_new, K_new, dist_coefs_new


def save_bundle_rd_out(structure, K,
                       rotations, motions,
                       colors,
                       correspond_struct_idxs,
                       keypoints_for_all):
    """
    :param structure:
    :param K:
    :param rotations:
    :param motions:
    :param colors:
    :param correspond_struct_idxs:
    :param keypoints_for_all:
    :return:
    """
    # print("处理数据")
    lines = []
    # header = "# Bundle file v0.3"
    lines.append("# Bundle file v0.3")
    n_cams = len(rotations)
    n_pts = len(structure)
    lines.append(str(n_cams) + " " + str(n_pts))

    # print('1')
    for i in range(n_cams):
        focal_length = 0.5 * (K[0][0] + K[1][1])
        R = rotations[i]
        T = motions[i]
        lines.append(str(focal_length) + " 0 0")
        lines.append(str(R[0][0]) + " " + str(R[0][1]) + " " + str(R[0][2]))
        lines.append(str(R[1][0]) + " " + str(R[1][1]) + " " + str(R[1][2]))
        lines.append(str(R[2][0]) + " " + str(R[2][1]) + " " + str(R[2][2]))
        lines.append(str(T[0][0]) + " " + str(T[1][0]) + " " + str(T[2][0]))

    # print("2")
    for i in range(n_pts):
        lines.append(str(structure[i][0]) + " " + str(structure[i][1]) + " " + str(structure[i][2]))
        lines.append(str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]))

        # get detailed information of points
        # 1. the number of cameras in which can observe the point
        # 2. the corresponding information about projection points of the point in images  camera index, keypoint index, x/y coordinante
        count = 0  # counter the number of camera
        info = ""
        for a in range(len(correspond_struct_idxs)):
            if i in correspond_struct_idxs[a]:
                for b in range(len(correspond_struct_idxs[a])):
                    if correspond_struct_idxs[a][b] == i:
                        count += 1
                        info = info + str(a) + " "
                        info = info + str(b) + " "
                        info = info + str(keypoints_for_all[a][b].pt[0]) + " " + str(
                            keypoints_for_all[a][b].pt[1]) + " "
                        break

        lines.append(str(count) + " " + info)

    with open("../reconstruction/dense/pmvs/bundle.rd.out", "w") as f:
        # with open("../dense/pmvs/bundle.rd.out", "w") as f:
        # TODO: GUI更改文件夹位置3
        print("开始写入...")
        s = "\n"
        f.write(s.join(lines))


def save_sparse():
    image_dir = rec_config.image_dir

    points = np.float32(np.load(image_dir + '/Structure.npy'))
    colors = np.load(image_dir + '/Colors.npy')
    filename = 'sparse.ply'
    points = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])

    np.savetxt(image_dir + '/' + filename, points, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        \n
        '''

    with open(image_dir + '/' + filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(points)))
        f.write(old)


def SFM():
    image_dir = rec_config.image_dir + '/'
    image_names = glob.glob(image_dir + '*.jpg')  # 读取图片本身的名字
    image_names = sorted(image_names)
    with open("../project_name.txt", "r") as f:
        project_name = f.read()

    # TODO: GUI更改文件夹位置1
    # with open("../../project_name.txt", "r") as f:
    #     project_name = f.read()

    K = np.load('../calibration/camera_params/' + project_name + '/K.npy')

    # TODO: GUI更改文件夹位置2
    # K = np.load('../../calibration/camera_params/' + project_name + '/K.npy')

    # 加载畸变系数数组
    params_dir = '../calibration/camera_params/' + project_name
    distort_coefs = np.load(params_dir + '/dist.npy')
    # distort_coefs = []
    print('Distortion coefficients:\n', distort_coefs)

    # 提取特征点、特征匹配
    print('提取所有图片特征点...')
    kts_all, descriptors_for_all, colors_for_all = extract_feathers(image_names)
    # print(colors_for_all)

    print('匹配所有图片特征...')
    matches_for_all = match_all_feather(kts_all, descriptors_for_all)
    for i in range(len(matches_for_all)):
        print(len(matches_for_all[i]), end=' ')

    # 初始化点云
    print('\n初始化点云...')
    pts3d, inds_2d_to_3d, colors, rots, mots, projections = init_structure(K,
                                                                           kts_all,
                                                                           colors_for_all,
                                                                           matches_for_all)
    print("初始化点云数目:", len(pts3d))

    # 增量方式添加剩余点云
    print('增量方式添加剩余点云...')
    for i in tqdm(range(1, len(matches_for_all))):
        # 获取第i幅图中匹配点的空间三维坐标，以及第i+1幅图匹配点的像素坐标

        obj_points, img_points = get_objpts_and_imgpts(matches_for_all[i],
                                                       inds_2d_to_3d[i],
                                                       pts3d,
                                                       kts_all[i + 1])
        # solvePnPRansac得到第i+1个相机的旋转和平移
        # 在python的opencv中solvePnPRansac函数的第一个参数长度需要大于7，否则会报错
        # 这里对小于7的点集做一个重复填充操作，

        if len(obj_points) < 7:
            while len(img_points) < 7:
                obj_points = np.append(obj_points, [obj_points[0]], axis=0)
                img_points = np.append(img_points, [img_points[0]], axis=0)

        # 得到第i+1幅图相机的旋转向量和位移矩阵
        _, r, T, _ = cv2.solvePnPRansac(obj_points, img_points, K, np.array([]))
        R, _ = cv2.Rodrigues(r)  # 将旋转向量转换为旋转矩阵
        rots.append(R)  # 存入旋转矩阵list
        mots.append(T)

        # 根据[R|T]进行重建
        p1, p2 = get_matched_points(kts_all[i], kts_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])

        new_structure, new_proj = reconstruction(K, rots[i], mots[i], R, T, p1, p2)

        projections.append(new_proj)

        # 点云融合
        pts3d, colors, inds_2d_to_3d[i], inds_2d_to_3d[i + 1] = fuse_structure(
            matches_for_all[i],
            inds_2d_to_3d[i],
            inds_2d_to_3d[i + 1],
            pts3d,
            new_structure,
            colors, c1
        )
        print("新生成点云数", len(new_structure), "第", i, "次融合后点云数", len(pts3d))

    print(len(pts3d))

    ## ---------- BA优化
    print('删除误差较大的点')
    pts3d = delete_error_point(rots, mots, K, inds_2d_to_3d, kts_all, pts3d)

    # 由于经过bundle_adjustment的structure，会产生一些空的点（实际代表的意思是已被删除）
    # 修改各图片中关键点的索引
    # 修改点云中的None为 -1
    for i in range(len(pts3d)):
        if math.isnan(pts3d[i][0]):
            pts3d[i] = -1

    # 修改各图片中的索引
    for view_i in range(len(inds_2d_to_3d)):
        for pt2d_j in range(len(inds_2d_to_3d[view_i])):
            if inds_2d_to_3d[view_i][pt2d_j] != -1:
                if pts3d[int(inds_2d_to_3d[view_i][pt2d_j])][0] == -1:
                    inds_2d_to_3d[view_i][pt2d_j] = -1
                else:
                    inds_2d_to_3d[view_i][pt2d_j] -= (
                            np.sum(pts3d[:int(inds_2d_to_3d[view_i][pt2d_j])] == -1) / 3)

    # 删除那些为空的点
    i = 0
    while i < len(pts3d):
        if pts3d[i][0] == -1:
            pts3d = np.delete(pts3d, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1

    # ----- 计算重投影误差: 是否考虑畸变
    reproj_err(rots, mots, K,
               inds_2d_to_3d, kts_all, pts3d,
               distort_coefs)

    ## ---------- BA优化
    print('\nBA优化...')
    mots = np.array(mots)
    rots = np.array(rots)
    pts3d_before = pts3d.copy()
    pts3d_, poses_, K_, distort_coefs_ = BA(pts3d_before,
                                            inds_2d_to_3d,
                                            mots, rots,
                                            kts_all,
                                            K, distort_coefs)
    ## ----------

    # ----- 计算重投影误差: 是否考虑畸变
    reproj_err(rots, mots, K_,
               inds_2d_to_3d, kts_all, pts3d_,
               distort_coefs_)

    # ----- 计算优化前后的差异值
    # 计算空间点坐标和相机内参及畸变系数差异
    print('K diff:\n', K_ - K)
    print('Distortion coefficients diff:\n', distort_coefs_ - distort_coefs)
    print('Pts3d diff:\n', pts3d_ - pts3d_before)

    # 计算位姿差异
    n_cams = len(mots)
    rots_, mots_ = poses_[:n_cams * 3], poses_[n_cams * 3:n_cams * 6]
    for i in range(n_cams):
        rot_ = rots_[i * 3: (i + 1) * 3]
        rot, _ = cv2.Rodrigues(rots[i])  # 旋转矩阵转向量
        rot = rot.squeeze()
        print('Rotation vector diff of view {:d}:\n'.format(i), rot_ - rot)

        mot_ = mots_[i * 3: (i + 1) * 3]
        mot = mots[i].squeeze()
        print('Motion vector diff of view {:d}:\n'.format(i), mot_ - mot)

    # ----- 输出优化结果
    # 保存Bundle.rd.out
    print("点云已生成，正在保存.out文件")

    # # 旋转向量转化为旋转矩阵
    # Rotations = np.empty((n_cams, 3, 3))
    # for i in range(n_cams):
    #     R, _ = cv2.Rodrigues(rots[i])
    #     Rotations[i] = R

    save_bundle_rd_out(pts3d, K, rots, mots, colors, inds_2d_to_3d, kts_all)

    np.save(image_dir + 'Structure', pts3d)
    np.save(image_dir + 'Colors', colors)
    np.save(image_dir + 'Projections', projections)

    # np.save(image_dir + 'Rotations', rotations)
    # np.save(image_dir + 'Motions', motions)
    # np.save(image_dir + 'Indexs', correspond_struct_idx)
    #
    # for i in range(len(keypoints_for_all)):
    #     keypoints_for_all[i] = cv2.KeyPoint_convert(keypoints_for_all[i])
    # np.save(image_dir + 'Keypoints_for_all', keypoints_for_all)
