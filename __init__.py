bl_info = {
    "name": "Mesh Cut Plan/Elevation",
    "author": "Luiz + Codex",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Mesh Cut",
    "description": "Camera-based plan/elevation cuts for any mesh object with SVG/DXF export",
    "category": "Import-Export",
}

import math
from pathlib import Path

import bpy
import gpu
import blf
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup, UIList
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_plane

_DRAW_HANDLE_VIEW = None
_DRAW_HANDLE_TEXT = None
_DRAW_DIMENSION_PREVIEW = {"active": False, "p1": None, "p2": None, "label": ""}


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _get_or_create_meshcut_camera_collection(scene: bpy.types.Scene):
    collection_name = "MeshCut_Cameras"
    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
    if scene.collection.children.get(col.name) is None:
        scene.collection.children.link(col)
    return col


def _get_or_create_meshcut_annotation_collection(scene: bpy.types.Scene):
    collection_name = "MeshCut_Annotations"
    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
    if scene.collection.children.get(col.name) is None:
        scene.collection.children.link(col)
    return col


def _create_annotation_empty(scene: bpy.types.Scene, name: str, location: Vector):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.15
    empty.location = location
    _get_or_create_meshcut_annotation_collection(scene).objects.link(empty)
    return empty


def _remove_annotation_object(obj_name: str):
    if not obj_name:
        return
    obj = bpy.data.objects.get(obj_name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def _tag_redraw_view3d() -> None:
    wm = bpy.context.window_manager
    for window in wm.windows:
        screen = window.screen
        if not screen:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _update_redraw(_self=None, _context=None):
    _tag_redraw_view3d()


def _camera_depth(local_point: Vector) -> float:
    return -local_point.z


def _clip_segment_depth(p1: Vector, p2: Vector, dmin: float, dmax: float):
    d1 = _camera_depth(p1)
    d2 = _camera_depth(p2)
    dd = d2 - d1
    t_enter = 0.0
    t_exit = 1.0

    if dd == 0.0:
        if d1 < dmin:
            return None
    else:
        t = (dmin - d1) / dd
        if dd > 0.0:
            t_enter = max(t_enter, t)
        else:
            t_exit = min(t_exit, t)

    if dd == 0.0:
        if d1 > dmax:
            return None
    else:
        t = (dmax - d1) / dd
        if dd > 0.0:
            t_exit = min(t_exit, t)
        else:
            t_enter = max(t_enter, t)

    if t_enter > t_exit:
        return None

    q1 = p1.lerp(p2, t_enter)
    q2 = p1.lerp(p2, t_exit)
    return q1, q2, t_enter, t_exit


def _project_point(camera: bpy.types.Object, local_point: Vector) -> tuple[float, float]:
    if camera.data.type == "PERSP":
        z = -local_point.z
        if z <= 1e-9:
            return 0.0, 0.0
        return local_point.x / z, local_point.y / z
    return local_point.x, local_point.y


def _resolve_export_context(context: bpy.types.Context):
    scene = getattr(context, "scene", None) or bpy.context.scene
    view_layer = getattr(context, "view_layer", None) or bpy.context.view_layer
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return scene, view_layer, depsgraph


def _create_performance_budget(settings):
    visible_only = bool(getattr(settings, "visible_only", True))
    fallback_visible = bool(getattr(settings, "budget_fallback_visible", True))
    return {
        "enabled": bool(getattr(settings, "performance_guard", False)),
        "max_ray_casts": int(getattr(settings, "max_ray_casts", 200000)),
        # "Visible Only (No X-Ray)" must never degrade to x-ray when guard limit is reached.
        "fallback_visible": fallback_visible and not visible_only,
        "ray_casts": 0,
        "limit_hit": False,
    }


def _build_visibility_cache(view_layer, depsgraph):
    entries = []
    for obj in view_layer.objects:
        if obj.type != "MESH":
            continue
        if not obj.visible_get(view_layer=view_layer):
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        bvh = BVHTree.FromObject(eval_obj, depsgraph, epsilon=0.0)
        if bvh is None:
            continue
        matrix_world = eval_obj.matrix_world.copy()
        entries.append(
            {
                "bvh": bvh,
                "matrix_world": matrix_world,
                "matrix_world_inv": matrix_world.inverted_safe(),
            }
        )
    return {"entries": entries}


def _bvh_cache_ray_cast_nearest_distance(visibility_cache, origin_world: Vector, direction_world: Vector, max_distance: float):
    nearest = None
    world_target = origin_world + direction_world * max_distance

    for entry in visibility_cache["entries"]:
        mat = entry["matrix_world"]
        inv = entry["matrix_world_inv"]
        origin_local = inv @ origin_world
        target_local = inv @ world_target
        ray_local = target_local - origin_local
        local_distance = ray_local.length
        if local_distance <= 1e-12:
            continue

        direction_local = ray_local / local_distance
        hit_loc, _hit_normal, _face_idx, _hit_dist = entry["bvh"].ray_cast(origin_local, direction_local, local_distance)
        if hit_loc is None:
            continue

        hit_world = mat @ hit_loc
        hit_distance = (hit_world - origin_world).length
        if nearest is None or hit_distance < nearest:
            nearest = hit_distance

    return nearest


def _is_world_point_visible(scene, depsgraph, camera, point_world: Vector, epsilon: float = 1e-4, budget=None, visibility_cache=None) -> bool:
    if budget and budget["enabled"]:
        if budget["ray_casts"] >= budget["max_ray_casts"]:
            budget["limit_hit"] = True
            return True if budget["fallback_visible"] else False
        budget["ray_casts"] += 1

    if camera.data.type == "ORTHO":
        cam_inv = camera.matrix_world.inverted()
        p_cam = cam_inv @ point_world
        dist = -p_cam.z
        if dist <= epsilon:
            return True

        origin = camera.matrix_world @ Vector((p_cam.x, p_cam.y, 0.0))
        direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    else:
        origin = camera.matrix_world.translation
        ray = point_world - origin
        dist = ray.length
        if dist <= epsilon:
            return True
        direction = ray / dist

    if visibility_cache is not None:
        hit_dist = _bvh_cache_ray_cast_nearest_distance(visibility_cache, origin, direction, dist + epsilon)
        if hit_dist is None:
            return True
        return hit_dist >= dist - epsilon

    hit, location, _normal, _index, _obj, _matrix = scene.ray_cast(depsgraph, origin, direction, distance=dist + epsilon)
    if not hit:
        return True

    hit_dist = (location - origin).length
    return hit_dist >= dist - epsilon


def _visible_intervals_on_segment(scene, depsgraph, camera, p1_world: Vector, p2_world: Vector, samples: int, budget=None, visibility_cache=None):
    if budget and budget["enabled"] and budget["limit_hit"] and budget["fallback_visible"]:
        return [(0.0, 1.0)]

    samples = max(2, samples)
    step = 1.0 / samples
    vis_flags = []

    for i in range(samples + 1):
        t = i * step
        vis_flags.append(_is_world_point_visible(scene, depsgraph, camera, p1_world.lerp(p2_world, t), budget=budget, visibility_cache=visibility_cache))
        if budget and budget["enabled"] and budget["limit_hit"] and budget["fallback_visible"]:
            return [(0.0, 1.0)]

    intervals = []
    t_start = None
    for i in range(samples + 1):
        visible = vis_flags[i]
        t_cur = i * step
        if visible and t_start is None:
            t_start = t_cur
        if (not visible or i == samples) and t_start is not None:
            t_end = t_cur if visible else max(0.0, t_cur - step)
            if t_end > t_start:
                intervals.append((t_start, t_end))
            t_start = None

    return intervals


def _camera_frame_corners_at_depth(camera, scene, depth: float):
    frame = camera.data.view_frame(scene=scene)
    corners_local = []

    if camera.data.type == "ORTHO":
        for c in frame:
            corners_local.append(Vector((c.x, c.y, -depth)))
    else:
        for c in frame:
            z = -c.z if abs(c.z) > 1e-9 else 1.0
            corners_local.append(c * (depth / z))

    mat = camera.matrix_world
    return [mat @ c for c in corners_local]


def _draw_loop_lines(coords, lines):
    for i in range(4):
        lines.extend((coords[i], coords[(i + 1) % 4]))


def _is_preview_camera_context(context, camera) -> bool:
    if not context or not context.region_data or not context.space_data:
        return False
    if context.space_data.type != "VIEW_3D":
        return False

    scene = context.scene
    settings = getattr(scene, "meshcut_settings", None)
    if settings is None:
        return False

    if settings.preview_only_camera_view:
        if context.region_data.view_perspective != "CAMERA":
            return False
        if scene.camera != camera:
            return False
    return True


def _draw_depth_overlay():
    context = bpy.context
    if not context or not context.scene:
        return

    scene = context.scene
    settings = getattr(scene, "meshcut_settings", None)
    if settings is None or not settings.show_depth_overlay:
        return

    camera = settings.camera_obj or scene.camera
    if not camera or camera.type != "CAMERA":
        return
    if not _is_preview_camera_context(context, camera):
        return

    dmin = max(0.0, min(settings.depth_near, settings.depth_far))
    dmax = max(settings.depth_near, settings.depth_far)

    near_corners = _camera_frame_corners_at_depth(camera, scene, dmin)
    far_corners = _camera_frame_corners_at_depth(camera, scene, dmax)

    near_lines = []
    far_lines = []
    side_lines = []
    _draw_loop_lines(near_corners, near_lines)
    _draw_loop_lines(far_corners, far_lines)
    for i in range(4):
        side_lines.extend((near_corners[i], far_corners[i]))

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")

    shader.bind()
    shader.uniform_float("color", (0.1, 1.0, 0.2, 0.9))
    batch_for_shader(shader, "LINES", {"pos": near_lines}).draw(shader)
    shader.uniform_float("color", (1.0, 0.6, 0.1, 0.9))
    batch_for_shader(shader, "LINES", {"pos": far_lines}).draw(shader)
    shader.uniform_float("color", (0.9, 0.9, 0.9, 0.35))
    batch_for_shader(shader, "LINES", {"pos": side_lines}).draw(shader)

    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _format_length(scene, length: float) -> str:
    unit = scene.unit_settings.length_unit
    if unit in {"MILLIMETERS", "MILLIMETRES"}:
        return f"{length * 1000.0:.1f} mm"
    if unit in {"CENTIMETERS", "CENTIMETRES"}:
        return f"{length * 100.0:.2f} cm"
    if unit in {"METERS", "METRES", "NONE"}:
        return f"{length:.3f} m"
    return f"{length:.3f}"


def _angle_radians(p1: Vector, pivot: Vector, p3: Vector):
    v1 = p1 - pivot
    v2 = p3 - pivot
    if v1.length <= 1e-9 or v2.length <= 1e-9:
        return None
    c = max(-1.0, min(1.0, v1.normalized().dot(v2.normalized())))
    return math.acos(c)


def _format_angle(p1: Vector, pivot: Vector, p3: Vector) -> str:
    angle = _angle_radians(p1, pivot, p3)
    if angle is None:
        return "0.00 deg"
    return f"{math.degrees(angle):.2f} deg"


def _angle_label_position(p1: Vector, pivot: Vector, p3: Vector) -> Vector:
    v1 = p1 - pivot
    v2 = p3 - pivot
    l1 = v1.length
    l2 = v2.length
    if l1 <= 1e-9 or l2 <= 1e-9:
        return pivot

    u1 = v1 / l1
    u2 = v2 / l2
    bis = u1 + u2
    if bis.length <= 1e-9:
        bis = u1

    radius = max(0.05, min(l1, l2) * 0.25)
    return pivot + bis.normalized() * radius


def _color_to_rgb255(color) -> tuple[int, int, int]:
    r = int(max(0.0, min(1.0, float(color[0]))) * 255.0)
    g = int(max(0.0, min(1.0, float(color[1]))) * 255.0)
    b = int(max(0.0, min(1.0, float(color[2]))) * 255.0)
    return r, g, b


def _color_to_svg(color) -> str:
    r, g, b = _color_to_rgb255(color)
    return f"#{r:02X}{g:02X}{b:02X}"


def _color_to_dxf_truecolor(color) -> int:
    r, g, b = _color_to_rgb255(color)
    return (r << 16) | (g << 8) | b


def _sync_annotation_points_from_objects(ann):
    if ann.p1_object_name:
        obj1 = bpy.data.objects.get(ann.p1_object_name)
        if obj1 is not None:
            ann.p1 = obj1.matrix_world.translation
    if ann.p2_object_name:
        obj2 = bpy.data.objects.get(ann.p2_object_name)
        if obj2 is not None:
            ann.p2 = obj2.matrix_world.translation
    if ann.p3_object_name:
        obj3 = bpy.data.objects.get(ann.p3_object_name)
        if obj3 is not None:
            ann.p3 = obj3.matrix_world.translation

def _collect_annotation_world_geometry(scene, depsgraph, camera, settings, dmin: float, dmax: float, budget=None, visibility_cache=None):
    cam_inv = camera.matrix_world.inverted()
    world_lines = []
    world_texts = []

    for ann in scene.meshcut_annotations:
        _sync_annotation_points_from_objects(ann)
        p1_world = Vector(ann.p1)
        p1_cam = cam_inv @ p1_world

        if ann.annotation_type == "TEXT":
            depth = _camera_depth(p1_cam)
            if dmin <= depth <= dmax:
                if (not settings.visible_only) or _is_world_point_visible(scene, depsgraph, camera, p1_world, budget=budget, visibility_cache=visibility_cache):
                    world_texts.append((p1_world, ann.text.strip() or ann.name or "Note"))
            continue

        if ann.annotation_type == "ANGLE":
            pivot_world = Vector(ann.p2)
            p3_world = Vector(ann.p3)
            pivot_cam = cam_inv @ pivot_world
            p3_cam = cam_inv @ p3_world

            clipped1 = _clip_segment_depth(pivot_cam, p1_cam, dmin, dmax)
            clipped2 = _clip_segment_depth(pivot_cam, p3_cam, dmin, dmax)
            any_line = False

            for clipped in (clipped1, clipped2):
                if clipped is None:
                    continue
                _c1_cam, _c2_cam, t0, t1 = clipped
                c1_world = pivot_world.lerp(p1_world if clipped is clipped1 else p3_world, t0)
                c2_world = pivot_world.lerp(p1_world if clipped is clipped1 else p3_world, t1)

                vis_parts = [(0.0, 1.0)]
                if settings.visible_only:
                    vis_parts = _visible_intervals_on_segment(
                        scene,
                        depsgraph,
                        camera,
                        c1_world,
                        c2_world,
                        settings.visibility_samples,
                        budget=budget,
                        visibility_cache=visibility_cache,
                    )

                for s0, s1 in vis_parts:
                    v1_world = c1_world.lerp(c2_world, s0)
                    v2_world = c1_world.lerp(c2_world, s1)
                    if (v2_world - v1_world).length > 1e-8:
                        world_lines.append((v1_world, v2_world))
                        any_line = True

            if any_line:
                label = ann.text.strip() or _format_angle(p1_world, pivot_world, p3_world)
                label_pos = _angle_label_position(p1_world, pivot_world, p3_world)
                depth = _camera_depth(cam_inv @ label_pos)
                if dmin <= depth <= dmax:
                    if (not settings.visible_only) or _is_world_point_visible(scene, depsgraph, camera, label_pos, budget=budget, visibility_cache=visibility_cache):
                        world_texts.append((label_pos, label))
            continue

        p2_world = Vector(ann.p2)
        p2_cam = cam_inv @ p2_world
        clipped = _clip_segment_depth(p1_cam, p2_cam, dmin, dmax)
        if clipped is None:
            continue

        c1_cam, c2_cam, t0, t1 = clipped
        c1_world = p1_world.lerp(p2_world, t0)
        c2_world = p1_world.lerp(p2_world, t1)

        vis_parts = [(0.0, 1.0)]
        if settings.visible_only:
            vis_parts = _visible_intervals_on_segment(
                scene,
                depsgraph,
                camera,
                c1_world,
                c2_world,
                settings.visibility_samples,
                budget=budget,
                visibility_cache=visibility_cache,
            )

        any_line = False
        for s0, s1 in vis_parts:
            v1_world = c1_world.lerp(c2_world, s0)
            v2_world = c1_world.lerp(c2_world, s1)
            if (v2_world - v1_world).length > 1e-8:
                world_lines.append((v1_world, v2_world))
                any_line = True

        if any_line:
            label = ann.text.strip() or _format_length(scene, (c2_world - c1_world).length)
            mid_world = (c1_world + c2_world) * 0.5
            if (not settings.visible_only) or _is_world_point_visible(scene, depsgraph, camera, mid_world, budget=budget, visibility_cache=visibility_cache):
                world_texts.append((mid_world, label))

    return world_lines, world_texts


def _collect_annotation_geometry(scene, depsgraph, camera, settings, dmin: float, dmax: float, budget=None, visibility_cache=None):
    cam_inv = camera.matrix_world.inverted()
    world_lines, world_texts = _collect_annotation_world_geometry(
        scene, depsgraph, camera, settings, dmin, dmax, budget=budget, visibility_cache=visibility_cache
    )
    anno_lines = []
    anno_texts = []

    for w1, w2 in world_lines:
        anno_lines.append((_project_point(camera, cam_inv @ w1), _project_point(camera, cam_inv @ w2)))

    for w, txt in world_texts:
        anno_texts.append((_project_point(camera, cam_inv @ w), txt))

    return anno_lines, anno_texts


def _collect_segments(context: bpy.types.Context):
    scene, view_layer, depsgraph = _resolve_export_context(context)
    settings = scene.meshcut_settings
    camera = settings.camera_obj or scene.camera
    if not camera or camera.type != "CAMERA":
        raise RuntimeError("Pick a valid camera in Mesh Cut > Camera.")

    if settings.use_selected_only:
        mesh_objects = [obj for obj in view_layer.objects if obj.type == "MESH" and obj.select_get(view_layer=view_layer)]
    else:
        mesh_objects = [obj for obj in view_layer.objects if obj.type == "MESH" and obj.visible_get(view_layer=view_layer)]

    if not mesh_objects and len(scene.meshcut_annotations) == 0:
        if settings.use_selected_only:
            raise RuntimeError("No selected mesh objects found. Disable 'Selected Only' or select meshes.")
        raise RuntimeError("No visible mesh objects found for export.")

    cam_inv = camera.matrix_world.inverted()
    segments = []
    points = []
    budget = _create_performance_budget(settings)
    visibility_cache = _build_visibility_cache(view_layer, depsgraph) if settings.visible_only else None

    dmin = min(settings.depth_near, settings.depth_far)
    dmax = max(settings.depth_near, settings.depth_far)

    for obj in mesh_objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue

        try:
            world_mat = eval_obj.matrix_world
            world_verts = [world_mat @ v.co for v in mesh.vertices]
            cam_verts = [cam_inv @ p for p in world_verts]

            if settings.export_vertices:
                for p_cam, p_world in zip(cam_verts, world_verts):
                    depth = _camera_depth(p_cam)
                    if dmin <= depth <= dmax:
                        if (not settings.visible_only) or _is_world_point_visible(
                            scene, depsgraph, camera, p_world, budget=budget, visibility_cache=visibility_cache
                        ):
                            points.append(_project_point(camera, p_cam))

            for edge in mesh.edges:
                i1 = edge.vertices[0]
                i2 = edge.vertices[1]
                p1_cam = cam_verts[i1]
                p2_cam = cam_verts[i2]
                p1_world = world_verts[i1]
                p2_world = world_verts[i2]

                clipped = _clip_segment_depth(p1_cam, p2_cam, dmin, dmax)
                if clipped is None:
                    continue

                c1_cam, c2_cam, t0, t1 = clipped
                c1_world = p1_world.lerp(p2_world, t0)
                c2_world = p1_world.lerp(p2_world, t1)

                if not settings.visible_only:
                    segments.append((_project_point(camera, c1_cam), _project_point(camera, c2_cam)))
                    continue

                intervals = _visible_intervals_on_segment(
                    scene,
                    depsgraph,
                    camera,
                    c1_world,
                    c2_world,
                    settings.visibility_samples,
                    budget=budget,
                    visibility_cache=visibility_cache,
                )
                for s0, s1 in intervals:
                    v1_cam = c1_cam.lerp(c2_cam, s0)
                    v2_cam = c1_cam.lerp(c2_cam, s1)
                    if (v2_cam - v1_cam).length > 1e-8:
                        segments.append((_project_point(camera, v1_cam), _project_point(camera, v2_cam)))
        finally:
            eval_obj.to_mesh_clear()

    anno_lines, anno_texts = _collect_annotation_geometry(
        scene, depsgraph, camera, settings, dmin, dmax, budget=budget, visibility_cache=visibility_cache
    )

    if not segments and not points and not anno_lines and not anno_texts:
        raise RuntimeError("Nothing to export with current depth settings.")

    return segments, points, anno_lines, anno_texts, budget


def _draw_annotations_overlay_view():
    context = bpy.context
    if not context or not context.scene:
        return

    scene = context.scene
    settings = getattr(scene, "meshcut_settings", None)
    if settings is None or not settings.show_annotation_preview:
        return

    camera = settings.camera_obj or scene.camera
    if not camera or camera.type != "CAMERA":
        return
    if not _is_preview_camera_context(context, camera):
        return

    depsgraph = context.evaluated_depsgraph_get()
    dmin = min(settings.depth_near, settings.depth_far)
    dmax = max(settings.depth_near, settings.depth_far)
    world_lines, _world_texts = _collect_annotation_world_geometry(scene, depsgraph, camera, settings, dmin, dmax)

    if not world_lines and not _DRAW_DIMENSION_PREVIEW["active"]:
        return

    line_vertices = []
    for w1, w2 in world_lines:
        line_vertices.extend((w1, w2))

    if _DRAW_DIMENSION_PREVIEW["active"] and _DRAW_DIMENSION_PREVIEW["p1"] and _DRAW_DIMENSION_PREVIEW["p2"]:
        line_vertices.extend((_DRAW_DIMENSION_PREVIEW["p1"], _DRAW_DIMENSION_PREVIEW["p2"]))

    if not line_vertices:
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE" if settings.annotation_on_top else "LESS_EQUAL")
    shader.bind()
    shader.uniform_float("color", tuple(settings.annotation_line_color))
    batch_for_shader(shader, "LINES", {"pos": line_vertices}).draw(shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _draw_annotations_overlay_text():
    context = bpy.context
    if not context or not context.scene or not context.region or not context.region_data:
        return

    scene = context.scene
    settings = getattr(scene, "meshcut_settings", None)
    if settings is None or not settings.show_annotation_preview:
        return

    camera = settings.camera_obj or scene.camera
    if not camera or camera.type != "CAMERA":
        return
    if not _is_preview_camera_context(context, camera):
        return

    depsgraph = context.evaluated_depsgraph_get()
    dmin = min(settings.depth_near, settings.depth_far)
    dmax = max(settings.depth_near, settings.depth_far)
    _world_lines, world_texts = _collect_annotation_world_geometry(scene, depsgraph, camera, settings, dmin, dmax)

    font_id = 0
    blf.size(font_id, int(max(8, settings.viewport_text_size)))
    blf.color(font_id, *tuple(settings.annotation_text_color))

    for wpos, txt in world_texts:
        p2d = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, wpos)
        if p2d:
            blf.position(font_id, p2d.x + 4.0, p2d.y + 4.0, 0)
            blf.draw(font_id, txt)

    if _DRAW_DIMENSION_PREVIEW["active"] and _DRAW_DIMENSION_PREVIEW["p1"] and _DRAW_DIMENSION_PREVIEW["p2"]:
        mid = (_DRAW_DIMENSION_PREVIEW["p1"] + _DRAW_DIMENSION_PREVIEW["p2"]) * 0.5
        p2d = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, mid)
        if p2d:
            blf.position(font_id, p2d.x + 4.0, p2d.y + 4.0, 0)
            blf.draw(font_id, _DRAW_DIMENSION_PREVIEW["label"])


def _draw_view_overlay():
    _draw_depth_overlay()
    _draw_annotations_overlay_view()

def _bounds_2d(segments, points, anno_lines, anno_texts):
    xs = []
    ys = []

    for (x1, y1), (x2, y2) in segments:
        xs.extend((x1, x2))
        ys.extend((y1, y2))

    for (x1, y1), (x2, y2) in anno_lines:
        xs.extend((x1, x2))
        ys.extend((y1, y2))

    for x, y in points:
        xs.append(x)
        ys.append(y)

    for (x, y), _txt in anno_texts:
        xs.append(x)
        ys.append(y)

    if not xs:
        return 0.0, 0.0, 1.0, 1.0

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    if math.isclose(min_x, max_x):
        max_x += 1.0
    if math.isclose(min_y, max_y):
        max_y += 1.0

    return min_x, min_y, max_x, max_y


def _write_svg(filepath: str, segments, points, anno_lines, anno_texts, scale: float, margin: float, text_size: float, anno_line_color, anno_text_color):
    min_x, min_y, max_x, max_y = _bounds_2d(segments, points, anno_lines, anno_texts)
    width = (max_x - min_x) * scale + (2.0 * margin)
    height = (max_y - min_y) * scale + (2.0 * margin)

    def map_xy(x, y):
        sx = (x - min_x) * scale + margin
        sy = (max_y - y) * scale + margin
        return sx, sy

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width:.4f}" height="{height:.4f}" viewBox="0 0 {width:.4f} {height:.4f}">')
    lines.append('<g stroke="black" fill="none" stroke-width="1">')

    for (x1, y1), (x2, y2) in segments:
        sx1, sy1 = map_xy(x1, y1)
        sx2, sy2 = map_xy(x2, y2)
        lines.append(f'<line x1="{sx1:.4f}" y1="{sy1:.4f}" x2="{sx2:.4f}" y2="{sy2:.4f}" />')

    lines.append("</g>")

    if points:
        lines.append('<g fill="red" stroke="none">')
        for x, y in points:
            sx, sy = map_xy(x, y)
            lines.append(f'<circle cx="{sx:.4f}" cy="{sy:.4f}" r="1.5" />')
        lines.append("</g>")

    if anno_lines:
        stroke = _color_to_svg(anno_line_color)
        stroke_opacity = max(0.0, min(1.0, float(anno_line_color[3])))
        lines.append(f'<g stroke="{stroke}" stroke-opacity="{stroke_opacity:.4f}" fill="none" stroke-width="1.2">')
        for (x1, y1), (x2, y2) in anno_lines:
            sx1, sy1 = map_xy(x1, y1)
            sx2, sy2 = map_xy(x2, y2)
            lines.append(f'<line x1="{sx1:.4f}" y1="{sy1:.4f}" x2="{sx2:.4f}" y2="{sy2:.4f}" />')
        lines.append("</g>")

    if anno_texts:
        fill = _color_to_svg(anno_text_color)
        fill_opacity = max(0.0, min(1.0, float(anno_text_color[3])))
        lines.append(f'<g fill="{fill}" fill-opacity="{fill_opacity:.4f}" stroke="none">')
        for (x, y), txt in anno_texts:
            sx, sy = map_xy(x, y)
            safe_txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f'<text x="{sx:.4f}" y="{sy:.4f}" font-size="{text_size:.2f}">{safe_txt}</text>')
        lines.append("</g>")

    lines.append("</svg>")
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")

def _write_dxf(filepath: str, segments, points, anno_lines, anno_texts, text_height: float, anno_line_color, anno_text_color):
    lines = ["0", "SECTION", "2", "HEADER", "0", "ENDSEC", "0", "SECTION", "2", "TABLES", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    line_truecolor = str(_color_to_dxf_truecolor(anno_line_color))
    text_truecolor = str(_color_to_dxf_truecolor(anno_text_color))

    for (x1, y1), (x2, y2) in segments:
        lines.extend(["0", "LINE", "8", "0", "10", f"{x1:.6f}", "20", f"{y1:.6f}", "30", "0.0", "11", f"{x2:.6f}", "21", f"{y2:.6f}", "31", "0.0"])

    for (x1, y1), (x2, y2) in anno_lines:
        lines.extend(["0", "LINE", "8", "ANNOTATIONS", "420", line_truecolor, "10", f"{x1:.6f}", "20", f"{y1:.6f}", "30", "0.0", "11", f"{x2:.6f}", "21", f"{y2:.6f}", "31", "0.0"])

    for x, y in points:
        lines.extend(["0", "POINT", "8", "0", "10", f"{x:.6f}", "20", f"{y:.6f}", "30", "0.0"])

    for (x, y), txt in anno_texts:
        lines.extend(["0", "TEXT", "8", "ANNOTATIONS", "420", text_truecolor, "10", f"{x:.6f}", "20", f"{y:.6f}", "30", "0.0", "40", f"{text_height:.6f}", "1", txt])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    Path(filepath).write_text("\n".join(lines), encoding="ascii", errors="ignore")


def _get_dimension_points_from_context(context: bpy.types.Context):
    obj = context.active_object

    if context.mode == "EDIT_MESH" and obj and obj.type == "MESH":
        import bmesh

        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v for v in bm.verts if v.select]
        if len(verts) == 2:
            return obj.matrix_world @ verts[0].co, obj.matrix_world @ verts[1].co, None

    selected = [o for o in context.selected_objects if o.type in {"MESH", "EMPTY", "CURVE", "FONT", "LIGHT", "CAMERA"}]
    if len(selected) >= 2:
        return selected[0].matrix_world.translation.copy(), selected[1].matrix_world.translation.copy(), None

    return None, None, "Select 2 vertices in Edit Mode or 2 objects in Object Mode."


def _get_angle_points_from_context(context: bpy.types.Context):
    obj = context.active_object

    if context.mode == "EDIT_MESH" and obj and obj.type == "MESH":
        import bmesh

        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v for v in bm.verts if v.select]
        if len(verts) == 3:
            return (
                obj.matrix_world @ verts[0].co,
                obj.matrix_world @ verts[1].co,
                obj.matrix_world @ verts[2].co,
                None,
            )

    selected = [o for o in context.selected_objects if o.type in {"MESH", "EMPTY", "CURVE", "FONT", "LIGHT", "CAMERA"}]
    if len(selected) >= 3:
        return selected[0].matrix_world.translation.copy(), selected[1].matrix_world.translation.copy(), selected[2].matrix_world.translation.copy(), None

    return None, None, None, "Select 3 vertices in Edit Mode or 3 objects in Object Mode."


class MESHCUT_PG_annotation(PropertyGroup):
    annotation_type: EnumProperty(name="Type", items=[("TEXT", "Text", "Text note"), ("DIMENSION", "Dimension", "Distance dimension"), ("ANGLE", "Angle", "Angle dimension with 3 points")], default="TEXT")
    text: StringProperty(name="Text", default="")
    p1: FloatVectorProperty(name="P1", size=3, subtype="TRANSLATION", default=(0.0, 0.0, 0.0))
    p2: FloatVectorProperty(name="P2", size=3, subtype="TRANSLATION", default=(0.0, 0.0, 0.0))
    p3: FloatVectorProperty(name="P3", size=3, subtype="TRANSLATION", default=(0.0, 0.0, 0.0))
    p1_object_name: StringProperty(name="P1 Object", default="")
    p2_object_name: StringProperty(name="P2 Object", default="")
    p3_object_name: StringProperty(name="P3 Object", default="")


class MESHCUT_PG_settings(PropertyGroup):
    use_selected_only: BoolProperty(name="Selected Only", description="Export only selected mesh objects", default=True)
    export_vertices: BoolProperty(name="Export Vertices", description="Include vertices as points in SVG and DXF", default=True)
    visible_only: BoolProperty(name="Visible Only (No X-Ray)", description="Cast rays from camera and keep only directly visible geometry", default=True)
    performance_guard: BoolProperty(name="Performance Guard", description="Limit ray-casts to avoid Blender freezes on heavy scenes", default=True)
    max_ray_casts: IntProperty(name="Max Ray Casts", description="Maximum visibility tests per export", default=150000, min=1000, max=10000000, soft_max=1000000)
    budget_fallback_visible: BoolProperty(name="Fallback To Visible", description="When limit is reached, keep remaining geometry visible instead of skipping (ignored when Visible Only is enabled)", default=False)
    visibility_samples: IntProperty(
        name="Visibility Samples",
        description="Samples per edge for hidden-line filtering (higher = cleaner, slower)",
        default=64,
        min=2,
        max=2048,
        soft_max=512,
    )
    show_depth_overlay: BoolProperty(name="Show Near/Far Overlay", description="Draw depth range box in viewport", default=True, update=_update_redraw)
    depth_near: FloatProperty(name="Depth Near", description="Minimum depth from camera", default=0.0, min=0.0, soft_max=1000.0, update=_update_redraw)
    depth_far: FloatProperty(name="Depth Far", description="Maximum depth from camera", default=100.0, min=0.0, soft_max=1000.0, update=_update_redraw)
    svg_scale: FloatProperty(name="SVG Scale", description="SVG scale", default=100.0, min=0.001, soft_max=10000.0)
    svg_margin: FloatProperty(name="SVG Margin", description="SVG margin", default=20.0, min=0.0, soft_max=1000.0)
    svg_text_size: FloatProperty(name="SVG Text Size", description="Text size in SVG pixels", default=12.0, min=1.0, soft_max=200.0)
    dxf_text_height: FloatProperty(name="DXF Text Height", description="Text height in DXF units", default=0.2, min=0.0001, soft_max=1000.0)
    new_text_label: StringProperty(name="New Text", description="Text used for new note", default="Note")
    new_dim_label: StringProperty(name="Dimension Label", description="Custom label for new dimensions (empty = auto length)", default="")
    dimension_default_length: FloatProperty(name="Dimension Length", description="Default length when creating a new dimension object", default=1.0, min=0.001, soft_max=1000.0)
    annotation_on_top: BoolProperty(name="Annotations On Top", description="Draw annotation lines over geometry in viewport", default=True, update=_update_redraw)
    annotation_line_color: FloatVectorProperty(name="Annotation Line Color", description="Line color for annotation preview and export", subtype="COLOR", size=4, min=0.0, max=1.0, default=(1.0, 0.85, 0.1, 0.95), update=_update_redraw)
    annotation_text_color: FloatVectorProperty(name="Annotation Text Color", description="Text color for annotation preview and export", subtype="COLOR", size=4, min=0.0, max=1.0, default=(1.0, 0.95, 0.75, 1.0), update=_update_redraw)
    show_annotation_preview: BoolProperty(name="Show Annotation Preview", description="Show dimensions and notes in viewport", default=True, update=_update_redraw)
    preview_only_camera_view: BoolProperty(name="Preview Only Camera View", description="Show preview only when looking through the configured camera", default=True, update=_update_redraw)
    viewport_text_size: IntProperty(name="Viewport Text Size", description="Text size for viewport preview", default=14, min=8, max=72)
    camera_obj: PointerProperty(name="Camera", type=bpy.types.Object, description="Projection camera", poll=lambda _self, obj: obj.type == "CAMERA", update=_update_redraw)


class MESHCUT_UL_annotations(UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        icon = "FONT_DATA"
        if item.annotation_type == "DIMENSION":
            icon = "DRIVER_DISTANCE"
        elif item.annotation_type == "ANGLE":
            icon = "DRIVER_ROTATIONAL_DIFFERENCE"
        layout.label(text=(item.text if item.text else item.name), icon=icon)


class MESHCUT_OT_add_text_annotation(Operator):
    bl_idname = "meshcut.add_text_annotation"
    bl_label = "Add Text Annotation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.meshcut_settings
        ann = scene.meshcut_annotations.add()
        ann.annotation_type = "TEXT"
        ann.text = settings.new_text_label.strip() or "Note"
        ann.name = f"Text {len(scene.meshcut_annotations)}"
        cursor = scene.cursor.location.copy()
        ann.p1 = cursor
        ann.p2 = cursor
        ann.p3 = cursor
        obj = _create_annotation_empty(scene, f"MC_Text_{len(scene.meshcut_annotations):03d}", cursor)
        ann.p1_object_name = obj.name
        ann.p2_object_name = ""
        ann.p3_object_name = ""
        scene.meshcut_annotation_index = len(scene.meshcut_annotations) - 1
        self.report({"INFO"}, "Text annotation created.")
        return {"FINISHED"}


class MESHCUT_OT_add_dimension_annotation(Operator):
    bl_idname = "meshcut.add_dimension_annotation"
    bl_label = "Add Dimension Object"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.meshcut_settings
        cursor = scene.cursor.location.copy()
        p1 = cursor
        p2 = cursor + Vector((settings.dimension_default_length, 0.0, 0.0))

        ann = scene.meshcut_annotations.add()
        ann.annotation_type = "DIMENSION"
        ann.text = settings.new_dim_label.strip()
        ann.name = f"Dim {len(scene.meshcut_annotations)}"
        ann.p1 = p1
        ann.p2 = p2
        ann.p3 = p2
        obj1 = _create_annotation_empty(scene, f"MC_Dim_A_{len(scene.meshcut_annotations):03d}", Vector(p1))
        obj2 = _create_annotation_empty(scene, f"MC_Dim_B_{len(scene.meshcut_annotations):03d}", Vector(p2))
        ann.p1_object_name = obj1.name
        ann.p2_object_name = obj2.name
        ann.p3_object_name = ""
        scene.meshcut_annotation_index = len(scene.meshcut_annotations) - 1
        self.report({"INFO"}, "Dimension object created. Move A/B empties to set endpoints.")
        return {"FINISHED"}


class MESHCUT_OT_add_angle_annotation(Operator):
    bl_idname = "meshcut.add_angle_annotation"
    bl_label = "Add Angle Dimension"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.meshcut_settings
        p1, p2, p3, err = _get_angle_points_from_context(context)
        if err:
            cursor = scene.cursor.location.copy()
            p1 = cursor + Vector((settings.dimension_default_length, 0.0, 0.0))
            p2 = cursor
            p3 = cursor + Vector((0.0, settings.dimension_default_length, 0.0))

        ann = scene.meshcut_annotations.add()
        ann.annotation_type = "ANGLE"
        ann.text = settings.new_dim_label.strip()
        ann.name = f"Angle {len(scene.meshcut_annotations)}"
        ann.p1 = p1
        ann.p2 = p2
        ann.p3 = p3
        obj1 = _create_annotation_empty(scene, f"MC_Ang_A_{len(scene.meshcut_annotations):03d}", Vector(p1))
        obj2 = _create_annotation_empty(scene, f"MC_Ang_B_{len(scene.meshcut_annotations):03d}", Vector(p2))
        obj3 = _create_annotation_empty(scene, f"MC_Ang_C_{len(scene.meshcut_annotations):03d}", Vector(p3))
        ann.p1_object_name = obj1.name
        ann.p2_object_name = obj2.name
        ann.p3_object_name = obj3.name
        scene.meshcut_annotation_index = len(scene.meshcut_annotations) - 1
        self.report({"INFO"}, "Angle dimension created. Move A/B/C empties (B is the vertex).")
        return {"FINISHED"}


class MESHCUT_OT_remove_annotation(Operator):
    bl_idname = "meshcut.remove_annotation"
    bl_label = "Remove Annotation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        idx = scene.meshcut_annotation_index
        if idx < 0 or idx >= len(scene.meshcut_annotations):
            self.report({"ERROR"}, "No annotation selected.")
            return {"CANCELLED"}
        ann = scene.meshcut_annotations[idx]
        _remove_annotation_object(ann.p1_object_name)
        _remove_annotation_object(ann.p2_object_name)
        _remove_annotation_object(ann.p3_object_name)
        scene.meshcut_annotations.remove(idx)
        scene.meshcut_annotation_index = min(idx, max(0, len(scene.meshcut_annotations) - 1))
        self.report({"INFO"}, "Annotation removed.")
        return {"FINISHED"}


class MESHCUT_OT_clear_annotations(Operator):
    bl_idname = "meshcut.clear_annotations"
    bl_label = "Clear Annotations"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        for ann in scene.meshcut_annotations:
            _remove_annotation_object(ann.p1_object_name)
            _remove_annotation_object(ann.p2_object_name)
            _remove_annotation_object(ann.p3_object_name)
        scene.meshcut_annotations.clear()
        scene.meshcut_annotation_index = 0
        self.report({"INFO"}, "All annotations cleared.")
        return {"FINISHED"}

class MESHCUT_OT_create_camera(Operator):
    bl_idname = "meshcut.create_camera"
    bl_label = "Create Mesh Cut Camera"
    bl_description = "Create an orthographic camera aligned for plan/elevation"
    bl_options = {"REGISTER", "UNDO"}

    view_type: EnumProperty(name="Type", items=[("PLAN", "Plan", "Top view"), ("ELEVATION_FRONT", "Elevation Front", "Front view"), ("ELEVATION_RIGHT", "Elevation Right", "Right view"), ("ELEVATION_LEFT", "Elevation Left", "Left view"), ("ELEVATION_BACK", "Elevation Back", "Back view")], default="PLAN")
    distance: FloatProperty(name="Distance", default=20.0, min=0.1, soft_max=10000.0)
    ortho_scale: FloatProperty(name="Ortho Scale", default=50.0, min=0.01, soft_max=10000.0)

    def execute(self, context):
        cam_data = bpy.data.cameras.new(name=f"MeshCut_{self.view_type}")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = self.ortho_scale
        cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
        cam_collection = _get_or_create_meshcut_camera_collection(context.scene)
        cam_collection.objects.link(cam_obj)

        center = context.scene.cursor.location.copy()
        if self.view_type == "PLAN":
            direction = Vector((0.0, 0.0, 1.0)); rotation = (0.0, 0.0, 0.0)
        elif self.view_type == "ELEVATION_FRONT":
            direction = Vector((0.0, -1.0, 0.0)); rotation = (math.radians(90.0), 0.0, 0.0)
        elif self.view_type == "ELEVATION_RIGHT":
            direction = Vector((1.0, 0.0, 0.0)); rotation = (math.radians(90.0), 0.0, math.radians(90.0))
        elif self.view_type == "ELEVATION_LEFT":
            direction = Vector((-1.0, 0.0, 0.0)); rotation = (math.radians(90.0), 0.0, math.radians(-90.0))
        else:
            direction = Vector((0.0, 1.0, 0.0)); rotation = (math.radians(90.0), 0.0, math.radians(180.0))

        cam_obj.location = center + direction * self.distance
        cam_obj.rotation_euler = rotation
        context.scene.camera = cam_obj
        context.scene.meshcut_settings.camera_obj = cam_obj
        _tag_redraw_view3d()
        self.report({"INFO"}, f"Camera created: {cam_obj.name}")
        return {"FINISHED"}


class MESHCUT_OT_export_svg(Operator):
    bl_idname = "meshcut.export_svg"
    bl_label = "Export SVG"
    bl_options = {"REGISTER"}
    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".svg"
    filter_glob: StringProperty(default="*.svg", options={"HIDDEN"})

    def execute(self, context):
        settings = (getattr(context, "scene", None) or bpy.context.scene).meshcut_settings
        try:
            segments, points, anno_lines, anno_texts, budget = _collect_segments(context)
            _write_svg(
                self.filepath,
                segments,
                points,
                anno_lines,
                anno_texts,
                settings.svg_scale,
                settings.svg_margin,
                settings.svg_text_size,
                settings.annotation_line_color,
                settings.annotation_text_color,
            )
            if budget["enabled"] and budget["limit_hit"]:
                mode = "fallback visibility" if budget["fallback_visible"] else "hidden remainder"
                self.report(
                    {"WARNING"},
                    f"Performance guard hit ({budget['ray_casts']} ray casts). Export mode: {mode}.",
                )
            self.report({"INFO"}, f"SVG exported: {self.filepath}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def invoke(self, context, _event):
        if not self.filepath:
            blend_name = Path(bpy.data.filepath).stem or "untitled"
            self.filepath = str(Path.home() / f"{_safe_name(blend_name)}_meshcut.svg")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class MESHCUT_OT_export_dxf(Operator):
    bl_idname = "meshcut.export_dxf"
    bl_label = "Export DXF"
    bl_options = {"REGISTER"}
    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".dxf"
    filter_glob: StringProperty(default="*.dxf", options={"HIDDEN"})

    def execute(self, context):
        settings = (getattr(context, "scene", None) or bpy.context.scene).meshcut_settings
        try:
            segments, points, anno_lines, anno_texts, budget = _collect_segments(context)
            _write_dxf(
                self.filepath,
                segments,
                points,
                anno_lines,
                anno_texts,
                settings.dxf_text_height,
                settings.annotation_line_color,
                settings.annotation_text_color,
            )
            if budget["enabled"] and budget["limit_hit"]:
                mode = "fallback visibility" if budget["fallback_visible"] else "hidden remainder"
                self.report(
                    {"WARNING"},
                    f"Performance guard hit ({budget['ray_casts']} ray casts). Export mode: {mode}.",
                )
            self.report({"INFO"}, f"DXF exported: {self.filepath}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def invoke(self, context, _event):
        if not self.filepath:
            blend_name = Path(bpy.data.filepath).stem or "untitled"
            self.filepath = str(Path.home() / f"{_safe_name(blend_name)}_meshcut.dxf")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class MESHCUT_PT_panel(Panel):
    bl_label = "Mesh Cut Plan/Elevation"
    bl_idname = "MESHCUT_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Cut"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.meshcut_settings

        col = layout.column(align=True)
        col.label(text="Camera")
        col.prop(settings, "camera_obj")

        row = col.row(align=True)
        op = row.operator("meshcut.create_camera", text="Plan"); op.view_type = "PLAN"
        op = row.operator("meshcut.create_camera", text="Front"); op.view_type = "ELEVATION_FRONT"

        row = col.row(align=True)
        op = row.operator("meshcut.create_camera", text="Right"); op.view_type = "ELEVATION_RIGHT"
        op = row.operator("meshcut.create_camera", text="Left"); op.view_type = "ELEVATION_LEFT"

        col.separator(); col.label(text="Depth Filter")
        col.prop(settings, "show_depth_overlay")
        col.prop(settings, "depth_near")
        col.prop(settings, "depth_far")

        col.separator(); col.label(text="Visibility")
        col.prop(settings, "visible_only")
        col.prop(settings, "performance_guard")
        if settings.performance_guard:
            col.prop(settings, "max_ray_casts")
            col.prop(settings, "budget_fallback_visible")
        col.prop(settings, "visibility_samples")

        col.separator(); col.label(text="Collection")
        col.prop(settings, "use_selected_only")
        col.prop(settings, "export_vertices")

        col.separator(); col.label(text="Annotations")
        col.prop(settings, "show_annotation_preview")
        col.prop(settings, "preview_only_camera_view")
        col.prop(settings, "annotation_on_top")
        col.prop(settings, "annotation_line_color")
        col.prop(settings, "annotation_text_color")
        col.prop(settings, "viewport_text_size")
        col.prop(settings, "new_text_label")
        col.operator("meshcut.add_text_annotation", icon="FONT_DATA")
        col.prop(settings, "new_dim_label")
        col.prop(settings, "dimension_default_length")
        col.operator("meshcut.add_dimension_annotation", icon="DRIVER_DISTANCE")
        col.operator("meshcut.add_angle_annotation", icon="DRIVER_ROTATIONAL_DIFFERENCE")
        col.template_list("MESHCUT_UL_annotations", "", scene, "meshcut_annotations", scene, "meshcut_annotation_index", rows=4)

        row = col.row(align=True)
        row.operator("meshcut.remove_annotation", icon="X")
        row.operator("meshcut.clear_annotations", icon="TRASH")

        col.separator(); col.label(text="Export")
        col.prop(settings, "svg_scale")
        col.prop(settings, "svg_margin")
        col.prop(settings, "svg_text_size")
        col.prop(settings, "dxf_text_height")
        col.operator("meshcut.export_svg", icon="EXPORT")
        col.operator("meshcut.export_dxf", icon="EXPORT")


classes = (
    MESHCUT_PG_annotation,
    MESHCUT_PG_settings,
    MESHCUT_UL_annotations,
    MESHCUT_OT_add_text_annotation,
    MESHCUT_OT_add_dimension_annotation,
    MESHCUT_OT_add_angle_annotation,
    MESHCUT_OT_remove_annotation,
    MESHCUT_OT_clear_annotations,
    MESHCUT_OT_create_camera,
    MESHCUT_OT_export_svg,
    MESHCUT_OT_export_dxf,
    MESHCUT_PT_panel,
)


def register():
    global _DRAW_HANDLE_VIEW, _DRAW_HANDLE_TEXT
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.meshcut_settings = PointerProperty(type=MESHCUT_PG_settings)
    bpy.types.Scene.meshcut_annotations = CollectionProperty(type=MESHCUT_PG_annotation)
    bpy.types.Scene.meshcut_annotation_index = IntProperty(default=0)

    if _DRAW_HANDLE_VIEW is None:
        _DRAW_HANDLE_VIEW = bpy.types.SpaceView3D.draw_handler_add(_draw_view_overlay, (), "WINDOW", "POST_VIEW")
    if _DRAW_HANDLE_TEXT is None:
        _DRAW_HANDLE_TEXT = bpy.types.SpaceView3D.draw_handler_add(_draw_annotations_overlay_text, (), "WINDOW", "POST_PIXEL")


def unregister():
    global _DRAW_HANDLE_VIEW, _DRAW_HANDLE_TEXT
    if _DRAW_HANDLE_VIEW is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE_VIEW, "WINDOW")
        _DRAW_HANDLE_VIEW = None
    if _DRAW_HANDLE_TEXT is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE_TEXT, "WINDOW")
        _DRAW_HANDLE_TEXT = None

    if hasattr(bpy.types.Scene, "meshcut_annotation_index"):
        del bpy.types.Scene.meshcut_annotation_index
    if hasattr(bpy.types.Scene, "meshcut_annotations"):
        del bpy.types.Scene.meshcut_annotations
    if hasattr(bpy.types.Scene, "meshcut_settings"):
        del bpy.types.Scene.meshcut_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
