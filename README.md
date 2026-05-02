# Mesh Cut Plan/Elevation

Blender add-on for plan, section, and elevation line extraction with SVG and DXF export.

Current release: `v2026.04.30`.

## Architecture

- `__init__.py`: main add-on entrypoint, UI, export flow, annotation tools, and visibility integration.
- `meshcut_accel.py`: BVH acceleration builder plus the visibility dispatcher that chooses Cython, Python BVH, or `scene.ray_cast`.
- `native_backend/`: package-scoped location for compiled extension modules shipped with the add-on.
- `cython/meshcut_parallel.pyx`: optional parallel ray-triangle kernel based on Moller-Trumbore intersection.
- `cython/setup.py`: build script for the optional `meshcut_parallel` extension.
- `blender_manifest.toml`: add-on manifest metadata.

## Visibility Pipeline

1. Mesh objects are evaluated through the depsgraph.
2. `meshcut_accel.py` triangulates those meshes and builds a world-space BVH with `BVHTree.FromPolygons`.
3. Export routines generate camera rays for points and sampled edge segments.
4. `visibility_mask(...)` resolves those rays in this order:
   - compiled Cython backend, when available;
   - Python BVH ray casts;
   - Blender `scene.ray_cast` fallback.
5. The exporter writes visible geometry and annotation data to SVG or DXF.

## Optional Cython Backend

The add-on works without the compiled backend, but production exports are faster with it.

### Linux or macOS

```bash
cd cython
python -m pip install cython setuptools wheel
python setup.py build_ext --inplace
```

### Windows

```powershell
cd cython
py -m pip install cython setuptools wheel
py setup.py build_ext --inplace
```

After the build completes, copy the generated `meshcut_parallel*.pyd` or `meshcut_parallel*.so` file into `native_backend/`.

## UI Options

The Visibility panel exposes the key runtime controls:

- `Visible Only (No X-Ray)`: enable hidden-line filtering from the active projection camera.
- `Require Cython Backend`: block export unless the compiled `meshcut_parallel` module is present.
- `Performance Guard`: cap the number of ray tests used by the export.
- `Always Finish Export`: when the ray budget runs out, keep the remaining geometry visible so the file still finishes exporting.
- `Fallback To Visible`: optional non-destructive fallback mode when the guard is hit outside the strict visible-only flow.
- `Visibility Samples`: edge sampling density for hidden-line filtering.

The Depth Filter panel also controls section output:

- `Show Dynamic Cuts`: draws live mesh intersections at `Depth Near` and `Depth Far` in the viewport.
- `Export Cut Edges`: creates real section lines from mesh faces crossing the near cut plane instead of relying only on existing mesh edges.
- `Export Cut Hatches`: creates hatch linework inside closed section contours.
- `Hatch Spacing`: distance between hatch lines.
- `Hatch Angle`: hatch direction in degrees.

The Annotations panel includes a viewport measurement tool:

- `Measure Mode`: creates linear, aligned, horizontal, vertical, or 3-point arc measurements.
- `Measure In Viewport`: click directly in the 3D Viewport to place a 2D camera-aligned measurement.

Viewport measurements are stored on the selected camera cut plane (`Depth Near`/`Depth Far` minimum) and are projected as 2D annotation geometry for preview plus SVG/DXF export.

The Export panel supports vector and viewport-shaded output:

- `Export SVG`: writes projected linework and annotations.
- `Export DXF`: writes layered CAD linework and annotations.
- `Export Camera View PNG`: writes the configured camera view using Blender's current OpenGL/viewport shading.

## DXF Layers

DXF export now writes entities to layers derived from object class names:

- projection edges and points: `<CLASS>`
- generated section lines: `<CLASS>_CORTE`
- generated hatch lines: `<CLASS>_HACHURA`
- annotation entities: `ANNOTATIONS`

The exporter detects common object-name keywords such as `parede`, `wall`, `projetor`, `projetro`, and `projeto`. If no known keyword exists, it uses the first object-name token before `.`, `_`, `-`, or a space.

## Recommended Production Preset

For the fastest and safest full export workflow:

- `Require Cython Backend` = ON
- `Performance Guard` = ON
- `Always Finish Export` = ON

This combination ensures the exporter prefers the compiled backend and still finishes the SVG or DXF even if the ray budget is exhausted.

## Performance Guard Behavior

When the guard is enabled, the add-on tracks a ray budget during export.

- If enough budget remains, segment visibility is tested in batches.
- If the remaining budget is low, the add-on automatically reduces the sample count instead of aborting.
- If the budget is fully exhausted and `Always Finish Export` is enabled, the remaining geometry is treated as visible so the export completes.

That means the add-on degrades occlusion quality gracefully instead of cancelling the output file.

## Validation Commands

```bash
python -m compileall __init__.py meshcut_accel.py cython/setup.py
rg -n "require_cython_backend|ensure_finish_on_budget|cython_backend_available|visibility_mask|BVHTree.FromPolygons" __init__.py meshcut_accel.py README.md
```
