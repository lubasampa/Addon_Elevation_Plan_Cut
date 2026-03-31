# Mesh Cut Plan/Elevation

Blender add-on for plan, section, and elevation line extraction with SVG and DXF export.

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
