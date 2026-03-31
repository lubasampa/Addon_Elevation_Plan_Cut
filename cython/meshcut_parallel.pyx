# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Parallel ray/triangle intersections for MeshCut."""

from cython.parallel import prange
from libc.math cimport fabs
from libc.stdlib cimport malloc, free
cimport cython


@cython.cfunc
cdef double _dot3(double ax, double ay, double az, double bx, double by, double bz) noexcept nogil:
    return ax * bx + ay * by + az * bz


@cython.cfunc
cdef bint _ray_triangle(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double v0x, double v0y, double v0z,
    double v1x, double v1y, double v1z,
    double v2x, double v2y, double v2z,
    double max_dist,
    double *hit_dist,
) noexcept nogil:
    cdef double e1x = v1x - v0x
    cdef double e1y = v1y - v0y
    cdef double e1z = v1z - v0z
    cdef double e2x = v2x - v0x
    cdef double e2y = v2y - v0y
    cdef double e2z = v2z - v0z

    cdef double px = dy * e2z - dz * e2y
    cdef double py = dz * e2x - dx * e2z
    cdef double pz = dx * e2y - dy * e2x
    cdef double det = _dot3(e1x, e1y, e1z, px, py, pz)

    if fabs(det) < 1e-12:
        return False

    cdef double inv_det = 1.0 / det
    cdef double tx = ox - v0x
    cdef double ty = oy - v0y
    cdef double tz = oz - v0z
    cdef double u = _dot3(tx, ty, tz, px, py, pz) * inv_det
    if u < 0.0 or u > 1.0:
        return False

    cdef double qx = ty * e1z - tz * e1y
    cdef double qy = tz * e1x - tx * e1z
    cdef double qz = tx * e1y - ty * e1x
    cdef double v = _dot3(dx, dy, dz, qx, qy, qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False

    cdef double t = _dot3(e2x, e2y, e2z, qx, qy, qz) * inv_det
    if t <= 0.0 or t > max_dist:
        return False

    hit_dist[0] = t
    return True


def parallel_first_hit_distances(origins, directions, distances, triangles):
    """Return nearest hit distance for each ray; -1 if no hit."""
    cdef Py_ssize_t nr = len(origins)
    cdef Py_ssize_t nt = len(triangles)
    cdef Py_ssize_t i, j, k

    cdef double *ray_o = <double *>malloc(sizeof(double) * nr * 3)
    cdef double *ray_d = <double *>malloc(sizeof(double) * nr * 3)
    cdef double *ray_max = <double *>malloc(sizeof(double) * nr)
    cdef double *tris = <double *>malloc(sizeof(double) * nt * 9)
    cdef double *hits = <double *>malloc(sizeof(double) * nr)

    cdef object ro, rd, tri
    cdef object v0, v1, v2
    cdef double max_d, best_d, hit_d

    if ray_o == cython.NULL or ray_d == cython.NULL or ray_max == cython.NULL or tris == cython.NULL or hits == cython.NULL:
        free(ray_o)
        free(ray_d)
        free(ray_max)
        free(tris)
        free(hits)
        raise MemoryError("Failed to allocate C buffers for ray tracing")

    try:
        for i in range(nr):
            ro = origins[i]
            rd = directions[i]
            ray_o[i * 3] = float(ro.x)
            ray_o[i * 3 + 1] = float(ro.y)
            ray_o[i * 3 + 2] = float(ro.z)
            ray_d[i * 3] = float(rd.x)
            ray_d[i * 3 + 1] = float(rd.y)
            ray_d[i * 3 + 2] = float(rd.z)
            ray_max[i] = float(distances[i])
            hits[i] = -1.0

        for i in range(nt):
            tri = triangles[i]
            v0 = tri[0]
            v1 = tri[1]
            v2 = tri[2]
            k = i * 9
            tris[k] = float(v0[0])
            tris[k + 1] = float(v0[1])
            tris[k + 2] = float(v0[2])
            tris[k + 3] = float(v1[0])
            tris[k + 4] = float(v1[1])
            tris[k + 5] = float(v1[2])
            tris[k + 6] = float(v2[0])
            tris[k + 7] = float(v2[1])
            tris[k + 8] = float(v2[2])

        with nogil:
            for i in prange(nr, schedule='guided'):
                max_d = ray_max[i]
                best_d = max_d + 1.0

                for j in range(nt):
                    k = j * 9
                    hit_d = 0.0
                    if _ray_triangle(
                        ray_o[i * 3], ray_o[i * 3 + 1], ray_o[i * 3 + 2],
                        ray_d[i * 3], ray_d[i * 3 + 1], ray_d[i * 3 + 2],
                        tris[k], tris[k + 1], tris[k + 2],
                        tris[k + 3], tris[k + 4], tris[k + 5],
                        tris[k + 6], tris[k + 7], tris[k + 8],
                        max_d,
                        &hit_d,
                    ):
                        if hit_d < best_d:
                            best_d = hit_d

                if best_d <= max_d:
                    hits[i] = best_d

        return [hits[i] for i in range(nr)]
    finally:
        free(ray_o)
        free(ray_d)
        free(ray_max)
        free(tris)
        free(hits)
