"""Acceleration layer for MeshCut visibility tests.

This module keeps the add-on runnable without compiled extensions, while
optionally using a Cython extension for parallel ray checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

try:
    # Compiled optional module (built from cython/meshcut_parallel.pyx).
    from . import meshcut_parallel as _cy_parallel
except Exception:  # pragma: no cover - Blender runtime dependent
    _cy_parallel = None


@dataclass
class VisibilityAcceleration:
    depsgraph: bpy.types.Depsgraph
    bvh: BVHTree | None
    tri_vertices: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]


def build_visibility_acceleration(mesh_objects: Iterable[bpy.types.Object], depsgraph: bpy.types.Depsgraph):
    """Build BVH and flattened triangle data from evaluated meshes."""
    vertices: list[tuple[float, float, float]] = []
    polygons: list[tuple[int, int, int]] = []
    tri_vertices: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

    for obj in mesh_objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue

        try:
            matrix = eval_obj.matrix_world
            base = len(vertices)

            for vert in mesh.vertices:
                world = matrix @ vert.co
                vertices.append((float(world.x), float(world.y), float(world.z)))

            mesh.calc_loop_triangles()
            for tri in mesh.loop_triangles:
                i0, i1, i2 = tri.vertices
                polygons.append((base + i0, base + i1, base + i2))
                tri_vertices.append((vertices[base + i0], vertices[base + i1], vertices[base + i2]))
        finally:
            eval_obj.to_mesh_clear()

    if not polygons:
        return VisibilityAcceleration(depsgraph=depsgraph, bvh=None, tri_vertices=[])

    bvh = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)
    return VisibilityAcceleration(depsgraph=depsgraph, bvh=bvh, tri_vertices=tri_vertices)


def visibility_mask(scene, accel: VisibilityAcceleration | None, origins: list[Vector], directions: list[Vector], distances: list[float], epsilon: float = 1e-4):
    """Return a visibility boolean for each ray.

    Uses Cython parallel kernel when available. Falls back to BVH ray_cast.
    """
    if not origins:
        return []

    if accel and accel.tri_vertices and _cy_parallel is not None:
        hit_distances = _cy_parallel.parallel_first_hit_distances(origins, directions, distances, accel.tri_vertices)
        return [hit < 0.0 or hit >= (dist - epsilon) for hit, dist in zip(hit_distances, distances)]

    bvh = accel.bvh if accel else None
    if bvh is not None:
        visible = []
        for origin, direction, dist in zip(origins, directions, distances):
            hit_loc, _normal, _index, _distance = bvh.ray_cast(origin, direction, dist + epsilon)
            if hit_loc is None:
                visible.append(True)
            else:
                hit_dist = (hit_loc - origin).length
                visible.append(hit_dist >= dist - epsilon)
        return visible

    visible = []
    depsgraph = accel.depsgraph if accel else bpy.context.evaluated_depsgraph_get()
    for origin, direction, dist in zip(origins, directions, distances):
        hit, location, _normal, _index, _obj, _matrix = scene.ray_cast(depsgraph, origin, direction, distance=dist + epsilon)
        if not hit:
            visible.append(True)
        else:
            visible.append((location - origin).length >= dist - epsilon)
    return visible
