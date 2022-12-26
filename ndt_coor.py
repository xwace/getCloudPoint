import open3d as o3d
import numpy as np
import os
import random
import copy
import cv2

from scipy import linalg



from scipy.spatial.transform import Rotation as R
# from handEye import get_rt
import pickle

# 将txt点云文件转换为o3d的pcd点云文件
def readtxt(name_path):
    pts = []
    with open(name_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            # 因为保存的TOF点云中，xy为图像面的xy，图像的值为深度距离z，因此转换到3维中，坐标为xzy
            pt = [float(line[0]), float(line[1]), float(line[2])]
            pts.append(pt)
    if len(pts) < 1000: return None # 小于1000个点的一帧点云直接抛弃掉
    # 点云保存到o3d中
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # 为点云设置随机颜色
    pcd.paint_uniform_color([random.random() for i in range(3)])  # 0, 0.651, 0.929 蓝色 # 1, 0.706, 0 黄色
    # o3d.io.write_point_cloud("621355968835992098.pcd", pcd)
    # o3d.visualization.draw_geometries([pcd], window_name="Open3D1")
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # 求点云的法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
    # 法向量对齐
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([1.0, 1.0, 1.0]))
    return pcd

# 参考 https://blog.csdn.net/u014072827/article/details/113788879
# 对点云进行下采样，同时输入的点云必须是已经完成法线估计的
# 最后计算每个点的FPFH特征。FPFH特征是一个33维向量，描述了一个点的局部几何特征。
# 在33维空间中的最近邻查询可以返回具有相似局部几何结构的点
def voxel_down(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=20))
    return pcd_down, pcd_fpfh

# 使用RANSAC进行全局配准。在RANSAC迭代中，每次从源点云中选取 ransac_n 个随机点，
# 通过33维FPFH特征空间中查询最近邻，可以检测到它们在目标点云中的对应点。
# 剪枝步骤需要使用快速剪枝算法来 提前 拒绝错误匹配：
#   CorrespondenceCheckerBasedOnDistance 检查对应的点云是否接近（也就是距离是否小于指定阈值）
#   CorrespondenceCheckerBasedOnEdgeLength 检查 source 和 target 中 分别绘制的任意两条边（由顶点形成的线）的长度是否相似。
#   CorrespondenceCheckerBasedOnNormal 考虑任何对应的顶点法线亲和力，它计算两个法向量的点积，它采用弧度值作为阈值
# 只有通过修剪步骤的匹配，才会用于计算变换，并在整个点云上进行验证，
# 核函数是 registration_ransac_based_on_feature_matching ，RANSACConvergenceCriteria 定义了 RANSAC 迭代的最大次数 和 验证的最大次数。
# 这两个值越大，那么结果越准确，但同时也花费更多的时间。
def global_registration(source, target):
    voxel_size = 0.01 # 下采样体素大小
    source_down, source_fpfh = voxel_down(source, voxel_size)
    target_down, target_fpfh = voxel_down(target, voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

# 局部优化匹配
# 出于性能方面的考虑，全局配准只在大量向下采样的点云上执行，配准的结果不够精细，
# 使用 TransformationEstimationPointToPlane 进一步优化配准结果
def ipc(source, target):
    # 进行RANSAC的全局配准
    result = global_registration(source, target)
    trans_init = result.transformation # 获取得到的全局配准矩阵作为icp局部匹配的初始值

    threshold = 0.01  # 移动范围的阀值
    # trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
    #                          [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
    #                          [0, 0, 1, 0],  # 这个矩阵为初始变换
    #                          [0, 0, 0, 1]])

    # 运行icp
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return reg_p2p

def get_matrix(tt):
    tt = np.array(tt)
    rotation = np.eye(4, dtype=float)
    Rm = R.from_quat(tt[3:])
    rotation_matrix = Rm.as_matrix()
    rotation[:3, :3] = rotation_matrix
    rotation[:3, 3] = tt[:3].T
    # print(rotation)
    return rotation

def read_pose(name_path):
    with open(name_path) as f:
        pts = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            pt = [float(i) for i in line]
            pts.append(get_matrix(pt))
    return pts

if __name__ == "__main__":
    # save_dir = "1205/"
    # name_dir = save_dir + "pcd/"
    # name_list = os.listdir(name_dir) # 获取当前文件夹下所有文件

    name_dir = './'
    name_list = os.listdir(os.getcwd()) # 获取当前文件夹下所有文件
    print(name_list)

    name_list = sorted(name_list, key=lambda x:float(x[:-8])) # 对文件名称进行降序，因为listdir获取得到的文件名不是按照降序获取的
    print(name_list)

    total_pcd = np.eye(4)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    last_pcd = readtxt(name_dir + name_list[0])  # 读取target文件

    pcds = [axis_pcd, last_pcd]

    for i in range(1, len(name_list)):
        print(i, ": ", name_list[i])
        # last_pcd = readtxt(name_dir + name_list[i-1]) # 读取target文件
        now_pcd = readtxt(name_dir + name_list[i]) # 读取source文件

        if last_pcd is not None and now_pcd is not None:
            now_pcd.transform(total_pcd)
            reg_p2p = ipc(now_pcd, pcds[-1]) # 进行icp匹配
            now_pcd.transform(reg_p2p.transformation)

            total_pcd = total_pcd@reg_p2p.transformation

            # print("tran = ", reg_p2p.transformation)
            # print("total_pcd = ", total_pcd)

            # print(reg_p2p.transformation)
            rr = np.asarray([[reg_p2p.transformation[0,0], reg_p2p.transformation[0,1], reg_p2p.transformation[0,2]],
                             [reg_p2p.transformation[1,0], reg_p2p.transformation[1,1], reg_p2p.transformation[1,2]],
                             [reg_p2p.transformation[2,0], reg_p2p.transformation[2,1], reg_p2p.transformation[2,2]]])
            #
            # # total_pcd = total_pcd@rr
            rr = R.from_matrix(rr) # 生成旋转矩阵
            rr = rr.as_euler('xyz', degrees=True)/180*np.pi # 将旋转矩阵 转换 成 欧拉角
            # # qua = rr.as_quat() # 将旋转矩阵 转换 成 四元数
            # # #
            print(rr)

            rr = R.from_matrix(total_pcd[:3, :3])  # 生成旋转矩阵
            rr = rr.as_euler('xyz', degrees=True) / 180 * np.pi  # 将旋转矩阵 转换 成 欧拉角
            # # qua = rr.as_quat() # 将旋转矩阵 转换 成 四元数
            # # #
            print(rr)

            # with open(save_dir + "pcd_diff/" + name_list[i], 'w') as file:
            #     sss = str(reg_p2p.transformation[0, 3]) + " " + str(reg_p2p.transformation[1, 3]) + " " + str(reg_p2p.transformation[2, 3]) + " " + \
            #           str(reg_p2p.transformation[0, 0]) + " " + str(reg_p2p.transformation[0, 1]) + " " + str(reg_p2p.transformation[0, 2]) + " " + \
            #           str(reg_p2p.transformation[1, 0]) + " " + str(reg_p2p.transformation[1, 1]) + " " + str(reg_p2p.transformation[1, 2]) + " " + \
            #           str(reg_p2p.transformation[2, 0]) + " " + str(reg_p2p.transformation[2, 1]) + " " + str(reg_p2p.transformation[2, 2])
            #     # print(sss)
            #     file.write(sss)

            # o3d.visualization.draw_geometries([axis_pcd, pcds[-1], now_pcd], window_name="Open3D1")
            pcds.append(now_pcd)
            # o3d.visualization.draw_geometries(pcds, window_name="Open3D1")
            # last_pcd = now_pcd

        print("########################################## \n")
    o3d.visualization.draw_geometries(pcds, window_name="Open3D1")




