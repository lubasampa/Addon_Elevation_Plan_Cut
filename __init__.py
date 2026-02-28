bl_info = {
    "name": "Projector Plan/Elevation Cut",
    "author": "Luiz + Codex",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Projector",
    "description": "Camera-based plan/elevation cuts for any mesh with SVG/DXF export",
    "category": "Import-Export",
}

import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _camera_depth(local_point: Vector) -> float:
    # In camera local coordinates, points in front of the camera are at negative Z.
    return -local_point.z


def _clip_segment_depth(p1: Vector, p2: Vector, dmin: float, dmax: float):
    d1 = _camera_depth(p1)
    d2 = _camera_depth(p2)
    dd = d2 - d1

    t_enter = 0.0
    t_exit = 1.0

    # Constraint: depth >= dmin
    if dd == 0.0:
        if d1 < dmin:
            return None
    else:
        t = (dmin - d1) / dd
        if dd > 0.0:
            t_enter = max(t_enter, t)
        else:
            t_exit = min(t_exit, t)

    # Constraint: depth <= dmax
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
    return q1, q2


def _project_point(camera: bpy.types.Object, local_point: Vector) -> tuple[float, float]:
    cam_data = camera.data
    if cam_data.type == "PERSP":
        z = -local_point.z
        if z <= 1e-9:
            return 0.0, 0.0
        return local_point.x / z, local_point.y / z
    return local_point.x, local_point.y


def _collect_segments(context: bpy.types.Context):
    settings = context.scene.projector_settings
    camera = settings.camera_obj or context.scene.camera
    if not camera or camera.type != "CAMERA":
        raise RuntimeError("Pick a valid camera in Projector > Camera.")

    if settings.use_selected_only:
        mesh_objects = [obj for obj in context.selected_objects if obj.type == "MESH"]
    else:
        mesh_objects = [obj for obj in context.visible_objects if obj.type == "MESH"]

    if not mesh_objects:
        raise RuntimeError("No mesh objects found for export.")

    depsgraph = context.evaluated_depsgraph_get()
    cam_inv = camera.matrix_world.inverted()

    segments = []
    points = []

    dmin = min(settings.depth_near, settings.depth_far)
    dmax = max(settings.depth_near, settings.depth_far)

    for obj in mesh_objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue

        try:
            mat = cam_inv @ eval_obj.matrix_world
            cam_verts = [mat @ v.co for v in mesh.vertices]

            if settings.export_vertices:
                for p in cam_verts:
                    depth = _camera_depth(p)
                    if dmin <= depth <= dmax:
                        points.append(_project_point(camera, p))

            for edge in mesh.edges:
                p1 = cam_verts[edge.vertices[0]]
                p2 = cam_verts[edge.vertices[1]]
                clipped = _clip_segment_depth(p1, p2, dmin, dmax)
                if clipped is None:
                    continue
                c1, c2 = clipped
                segments.append((_project_point(camera, c1), _project_point(camera, c2)))
        finally:
            eval_obj.to_mesh_clear()

    if not segments and not points:
        raise RuntimeError("Nothing to export with current depth settings.")

    return segments, points


def _bounds_2d(segments, points):
    xs = []
    ys = []

    for (x1, y1), (x2, y2) in segments:
        xs.extend((x1, x2))
        ys.extend((y1, y2))

    for x, y in points:
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


def _write_svg(filepath: str, segments, points, scale: float, margin: float):
    min_x, min_y, max_x, max_y = _bounds_2d(segments, points)
    width = (max_x - min_x) * scale + (2.0 * margin)
    height = (max_y - min_y) * scale + (2.0 * margin)

    def map_xy(x, y):
        sx = (x - min_x) * scale + margin
        sy = (max_y - y) * scale + margin
        return sx, sy

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width:.4f}" height="{height:.4f}" viewBox="0 0 {width:.4f} {height:.4f}">'
    )
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

    lines.append("</svg>")
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")


def _write_dxf(filepath: str, segments, points):
    lines = [
        "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
    ]

    for (x1, y1), (x2, y2) in segments:
        lines.extend(
            [
                "0", "LINE",
                "8", "0",
                "10", f"{x1:.6f}", "20", f"{y1:.6f}", "30", "0.0",
                "11", f"{x2:.6f}", "21", f"{y2:.6f}", "31", "0.0",
            ]
        )

    for x, y in points:
        lines.extend(
            [
                "0", "POINT",
                "8", "0",
                "10", f"{x:.6f}", "20", f"{y:.6f}", "30", "0.0",
            ]
        )

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    Path(filepath).write_text("\n".join(lines), encoding="ascii")


class PROJECTOR_PG_settings(PropertyGroup):
    use_selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected mesh objects",
        default=True,
    )
    export_vertices: BoolProperty(
        name="Export Vertices",
        description="Include vertices as points in SVG and DXF",
        default=True,
    )
    depth_near: FloatProperty(
        name="Depth Near",
        description="Minimum depth from camera",
        default=0.0,
        min=0.0,
        soft_max=1000.0,
    )
    depth_far: FloatProperty(
        name="Depth Far",
        description="Maximum depth from camera",
        default=100.0,
        min=0.0,
        soft_max=1000.0,
    )
    svg_scale: FloatProperty(
        name="SVG Scale",
        description="SVG scale (pixels per Blender unit)",
        default=100.0,
        min=0.001,
        soft_max=10000.0,
    )
    svg_margin: FloatProperty(
        name="SVG Margin",
        description="SVG margin in pixels",
        default=20.0,
        min=0.0,
        soft_max=1000.0,
    )
    camera_obj: PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="Projection camera (if empty, scene camera is used)",
        poll=lambda _self, obj: obj.type == "CAMERA",
    )


class PROJECTOR_OT_create_camera(Operator):
    bl_idname = "projector.create_camera"
    bl_label = "Create Projection Camera"
    bl_description = "Create an orthographic camera aligned for plan/elevation"
    bl_options = {"REGISTER", "UNDO"}

    view_type: EnumProperty(
        name="Type",
        items=[
            ("PLAN", "Plan", "Top view"),
            ("ELEVATION_FRONT", "Elevation Front", "Front view"),
            ("ELEVATION_RIGHT", "Elevation Right", "Right view"),
            ("ELEVATION_LEFT", "Elevation Left", "Left view"),
            ("ELEVATION_BACK", "Elevation Back", "Back view"),
        ],
        default="PLAN",
    )

    distance: FloatProperty(name="Distance", default=20.0, min=0.1, soft_max=10000.0)
    ortho_scale: FloatProperty(name="Ortho Scale", default=50.0, min=0.01, soft_max=10000.0)

    def execute(self, context):
        cam_data = bpy.data.cameras.new(name=f"Projector_{self.view_type}")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = self.ortho_scale

        cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
        context.collection.objects.link(cam_obj)

        center = context.scene.cursor.location.copy()

        if self.view_type == "PLAN":
            direction = Vector((0.0, 0.0, 1.0))
            rotation = (0.0, 0.0, 0.0)
        elif self.view_type == "ELEVATION_FRONT":
            direction = Vector((0.0, -1.0, 0.0))
            rotation = (math.radians(90.0), 0.0, 0.0)
        elif self.view_type == "ELEVATION_RIGHT":
            direction = Vector((1.0, 0.0, 0.0))
            rotation = (math.radians(90.0), 0.0, math.radians(90.0))
        elif self.view_type == "ELEVATION_LEFT":
            direction = Vector((-1.0, 0.0, 0.0))
            rotation = (math.radians(90.0), 0.0, math.radians(-90.0))
        else:
            direction = Vector((0.0, 1.0, 0.0))
            rotation = (math.radians(90.0), 0.0, math.radians(180.0))

        cam_obj.location = center + direction * self.distance
        cam_obj.rotation_euler = rotation

        context.scene.camera = cam_obj
        context.scene.projector_settings.camera_obj = cam_obj

        self.report({"INFO"}, f"Camera created: {cam_obj.name}")
        return {"FINISHED"}


class PROJECTOR_OT_export_svg(Operator):
    bl_idname = "projector.export_svg"
    bl_label = "Export SVG"
    bl_description = "Export current cut/projection to SVG"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".svg"
    filter_glob: StringProperty(default="*.svg", options={"HIDDEN"})

    def execute(self, context):
        settings = context.scene.projector_settings
        try:
            segments, points = _collect_segments(context)
            _write_svg(self.filepath, segments, points, settings.svg_scale, settings.svg_margin)
            self.report({"INFO"}, f"SVG exported: {self.filepath}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def invoke(self, context, _event):
        if not self.filepath:
            blend_name = Path(bpy.data.filepath).stem or "untitled"
            self.filepath = str(Path.home() / f"{_safe_name(blend_name)}_projector.svg")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class PROJECTOR_OT_export_dxf(Operator):
    bl_idname = "projector.export_dxf"
    bl_label = "Export DXF"
    bl_description = "Export current cut/projection to DXF"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".dxf"
    filter_glob: StringProperty(default="*.dxf", options={"HIDDEN"})

    def execute(self, context):
        try:
            segments, points = _collect_segments(context)
            _write_dxf(self.filepath, segments, points)
            self.report({"INFO"}, f"DXF exported: {self.filepath}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def invoke(self, context, _event):
        if not self.filepath:
            blend_name = Path(bpy.data.filepath).stem or "untitled"
            self.filepath = str(Path.home() / f"{_safe_name(blend_name)}_projector.dxf")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class PROJECTOR_PT_panel(Panel):
    bl_label = "Projector Plan/Elevation"
    bl_idname = "PROJECTOR_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Projector"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.projector_settings

        col = layout.column(align=True)
        col.label(text="Camera")
        col.prop(settings, "camera_obj")

        row = col.row(align=True)
        op = row.operator("projector.create_camera", text="Plan")
        op.view_type = "PLAN"
        op = row.operator("projector.create_camera", text="Front")
        op.view_type = "ELEVATION_FRONT"

        row = col.row(align=True)
        op = row.operator("projector.create_camera", text="Right")
        op.view_type = "ELEVATION_RIGHT"
        op = row.operator("projector.create_camera", text="Left")
        op.view_type = "ELEVATION_LEFT"

        col.separator()
        col.label(text="Depth Filter")
        col.prop(settings, "depth_near")
        col.prop(settings, "depth_far")

        col.separator()
        col.label(text="Collection")
        col.prop(settings, "use_selected_only")
        col.prop(settings, "export_vertices")

        col.separator()
        col.label(text="SVG")
        col.prop(settings, "svg_scale")
        col.prop(settings, "svg_margin")

        col.separator()
        col.operator("projector.export_svg", icon="EXPORT")
        col.operator("projector.export_dxf", icon="EXPORT")


classes = (
    PROJECTOR_PG_settings,
    PROJECTOR_OT_create_camera,
    PROJECTOR_OT_export_svg,
    PROJECTOR_OT_export_dxf,
    PROJECTOR_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.projector_settings = PointerProperty(type=PROJECTOR_PG_settings)


def unregister():
    del bpy.types.Scene.projector_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
