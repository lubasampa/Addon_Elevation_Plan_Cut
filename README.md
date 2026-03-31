# Mesh Cut Plan/Elevation (Blender Add-on)

Addon para gerar linhas técnicas de **planta/corte/elevação** com exportação SVG e DXF.

## Novidade nesta branch

Foi adicionada uma nova camada de aceleração para visibilidade/interseção de raios:

- Construção de BVH com `mathutils.bvhtree.BVHTree.FromPolygons` a partir de malhas avaliadas (`evaluated_get(depsgraph).to_mesh()`).
- Pipeline para receber e processar buffers de vértices/faces em coordenadas de mundo.
- Backend opcional em **Cython** (`cython/meshcut_parallel.pyx`) com kernel paralelo para teste de interseção raio-triângulo.
- Fallback automático para BVH em Python quando o módulo compilado não estiver disponível.

## Estrutura

- `__init__.py`: addon principal (UI, coleta de geometria, export SVG/DXF, integração com aceleração).
- `meshcut_accel.py`: construção BVH + despacho de backend de visibilidade (Cython/BVH/scene ray_cast).
- `cython/meshcut_parallel.pyx`: kernel de interseção de raios preparado para paralelismo.
- `cython/setup.py`: script de build do módulo Cython.

## Build do módulo Cython

> Recomendado compilar usando o Python do Blender para garantir ABI compatível.

### Linux/macOS

```bash
cd cython
python -m pip install cython setuptools wheel
python setup.py build_ext --inplace
```

### Windows (MSVC)

```powershell
cd cython
py -m pip install cython setuptools wheel
py setup.py build_ext --inplace
```

Após compilar, copie o binário gerado (`meshcut_parallel*.so`/`.pyd`) para a pasta raiz do addon (ao lado de `__init__.py`) para ativar o backend compilado.

## Fluxo de visibilidade

1. O addon coleta objetos de malha visíveis/selecionados.
2. Constrói um acelerador BVH com vértices/faces trianguladas.
3. Para cada ponto/amostra de segmento, gera raios da câmera.
4. Resolve a visibilidade:
   - **Cython** (se instalado): cálculo de primeiro hit em lote.
   - **BVH Python**: `bvh.ray_cast`.
   - **Fallback final**: `scene.ray_cast`.
5. Retorna segmentos/pontos prontos para serialização SVG/DXF.

## Compatibilidade e fallback

- O addon continua funcionando sem Cython.
- Quando o módulo Cython está ausente, a lógica usa BVH Python automaticamente.
- Exportações e ferramentas de anotação no viewport permanecem disponíveis.
