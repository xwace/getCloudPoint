# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/visualization.py

import numpy as np
# from open3d import *
import open3d as o3

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3.io.read_point_cloud("rabbit.pcd")

    pcd_tree = o3.geometry.KDTreeFlann(pcd)

    np.asarray(pcd.colors)[1500:1501,:] = [0.5, 0, 0.5]
    # pcd.colors = o3.utility.Vector3dVector(np.random.uniform(0, 1, size=(100, 3)))

    k = 5000
    [num, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500],k)
    np.asarray(pcd.colors)[1:, :] = [0, 0, 1]
    # np.asarray(pcd.colors)[400:600, 0:4] = [0, 0, 1]

    o3.visualization.draw_geometries([pcd])

    print("Let\'s draw some primitives")
    # mesh_box = o3.create_mesh_box(width = 1.0, height = 1.0, depth = 1.0)
    mesh_box = o3.geometry.TriangleMesh()
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_sphere = o3.geometry.TriangleMesh()
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = o3.geometry.TriangleMesh()
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = o3.geometry.TriangleMesh.create_coordinate_frame(size = 0.6, origin = [-2, -2, -2])

    print("We draw a few primitives using collection.")
    o3.visualization.draw_geometries([mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    o3.visualization.draw_geometries([mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

    print("Let\'s draw a cubic using LineSet")
    points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
              [0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    lines = [[0,1],[0,2],[1,3],[2,3],
             [4,5],[4,6],[5,7],[6,7],
             [0,4],[1,5],[2,6],[3,7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3.geometry.LineSet()
    line_set.points = o3.utility.Vector3dVector(points)
    line_set.lines = o3.utility.Vector2iVector(lines)
    line_set.colors = o3.utility.Vector3dVector(colors)
    o3.visualization.draw_geometries([line_set])
