import numpy
import scipy
import scipy.spatial


# 可以考虑retry一下
def sort(points):
    sorted_points = []

    # 用scipy的qhull寻找凸包，每次取出一个极点放入sorted_points。
    # 再用剩下的点进行循环，这样得到的序列反过来即满足要求
    last_index = None
    while(points.shape[0] > points.shape[1] + 1):
        # 用所有点得到hull
        hull = scipy.spatial.ConvexHull(points)
        index = numpy.random.choice(hull.vertices)

        # max_distance = 0
        # equation = None
        # index = -1
        # for p in hull.vertices:
        #     one_hull = scipy.spatial.ConvexHull(points, qhull_options="QG-{}".format(p))
        #     this_point = list(points[p])
        #     this_point.append(1)
        #     for e in one_hull.equations[numpy.logical_not(one_hull.good)]:
        #         distance = numpy.dot(this_point, e) / numpy.sqrt(numpy.dot(e[:-1], e[:-1]))
        #         if distance > max_distance:
        #             max_distance = distance
        #             equation = e
        #             index = p

        # if last_index is not N:
        #     min_dist = numpy.inf
        #
        #         p_point = points[p]
        #         last_point = points[last_index]
        #         dist = numpy.dot(last_point - p_point, last_point - p_point)
        #         if dist < min_dist:
        #             min_dist = dist
        #             index = p
        # last_index = index

        # 这里再调用了一次qhull，附加参数QG-n，表示去除的点n后的凸包，并可以得到点n对应的面
        hull = scipy.spatial.ConvexHull(
             points, qhull_options="QG-{}".format(index))
        # hull.good为False表示点n对应的面，默认去第0个了，有问题再改成随机
        # equation = hull.equations[numpy.logical_not(hull.good)][0]
        max_distance = 0
        select_equation = None
        this_points = list(points[index])
        this_points.append(1)
        for e in hull.equations[numpy.logical_not(hull.good)]:
            distance = numpy.dot(this_points, e) / numpy.sqrt(numpy.dot(e[:-1], e[:-1]))
            if distance > max_distance:
                max_distance = distance
                select_equation = e
        equation = select_equation
        # print(equation)
        # sorted_points里放入对应的点和该点和对应的方程
        sorted_points.append((points[index].copy(), equation))
        # print(sorted_points[-1], len(sorted_points))
        # 这边先把取出的点挪到最后，在去除，主要想避免拼接带来的性能影响，不确定有没有影响
        points[[points.shape[0] - 1, index]
               ] = points[[index, points.shape[0] - 1]]
        points = points[:-1]

    # 上述循环在点的数量为维度+1时退出，如果循环抛出异常应该是因为剩下的点有共线/共面的情况，重试一下就好
    # 剩下的维度+1个点，理论上是随便什么顺序都可以了，主要麻烦的是要求分界面方程，单独写了个surface函数来处理
    while(points.shape[0] > 1):
        # 每次取出最后一个点，并求最后一个点和其他点的分界面
        equation = surface(points[:-1], points[-1])
        sorted_points.append((points[-1], equation))
        # print(sorted_points[-1], len(sorted_points))
        points = points[:-1]
    # 最后一个点直接放进去，不需要方程
    sorted_points.append((points[0], None))
    # print(sorted_points[-1], len(sorted_points))
    # 倒序一下
    return list(reversed(sorted_points))


def surface(points, point):
    # points 包含若m点, point为一个点，向量维度为n，要求经过points且距离point最远的超平面
    # 基本思路是距离point最远的超平面的法向量应当是point到points向量的线性组合。
    x = points - point.reshape((1, -1))
    # 假设方程的法向量为a，那么a = alpha * x，其中alpha维度为m，是线性组合的系数。
    # 而法向量a和x又满足超平面的方程ax + b = 0，因此可以代入a得到下式
    #       alpha * (x * x^T) + b = 0
    # 该式是关于alpha的m元线性方程组，可以求出alpha，并进一步求出a。
    # 由于x肯定不经过0点（points和point的差），所以可以假定b=1来求解a，再归一化
    A = numpy.dot(x, x.transpose())
    B = numpy.ones((points.shape[0],))
    alpha = numpy.linalg.solve(A, B)
    a = numpy.dot(alpha, x)
    a /= numpy.sqrt((a ** 2).sum())  # 归一化
    b = -numpy.dot(a, points[0]).item()  # 任意代入一个点求b
    return numpy.array([*a, b])
