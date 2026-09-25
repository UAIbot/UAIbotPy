def obj_to_cpp(obj, htm=None):

    if get_uaibot_type(obj) != "uaibot.PointCloud" and htm is None:
        htm = obj.htm

    if get_uaibot_type(obj) == "uaibot.Ball":
        return ub_cpp.CPP_GeometricPrimitives.create_sphere(htm, obj.radius)
    if get_uaibot_type(obj) == "uaibot.Box":
        return ub_cpp.CPP_GeometricPrimitives.create_box(
            htm, obj.width, obj.depth, obj.height
        )
    if get_uaibot_type(obj) == "uaibot.Cylinder":
        return ub_cpp.CPP_GeometricPrimitives.create_cylinder(
            htm, obj.radius, obj.height
        )
    if get_uaibot_type(obj) == "uaibot.PointCloud":
        list_of_points = [
            obj._points[:, i : i + 1] for i in range(obj._points.shape[1])
        ]
        return ub_cpp.CPP_GeometricPrimitives.create_pointcloud(list_of_points)
    if get_uaibot_type(obj) == "uaibot.ConvexPolytope":
        Q = htm[0:3, 0:3]
        p = htm[0:3, -1]
        A = obj.A * Q.transpose()
        b = obj.b + obj.A * Q.transpose() * p
        return ub_cpp.CPP_GeometricPrimitives.create_convexpolytope(htm, A, b)
