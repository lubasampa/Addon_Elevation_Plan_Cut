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

## Evitando travamentos com muitos rays

Se a cena tiver muita geometria + `visibility_samples` alto, o número de raios cresce rápido.

Boas práticas:

- Ative `Performance Guard`.
- Comece com `Visibility Samples` entre **8 e 16**.
- Defina `Max Ray Casts` de forma conservadora (ex.: 50k ~ 200k).
- Para testes iniciais, desative `Visible Only` para validar corte/projeção sem o custo de ocultação.

Nesta branch o cálculo por segmento foi ajustado para:

- reduzir automaticamente amostras quando o budget restante é baixo;
- enviar os rays em lote para `visibility_mask` (menos overhead Python);
- parar cedo quando o limite do guard é atingido.

## Rodar backend Python fora do Blender

O addon completo depende de `bpy` e `mathutils`, então ele **não roda em Python puro padrão** sem ambiente Blender.

Você tem 2 caminhos:

1. **Recomendado: usar o Python do Blender**
   - Execute scripts com:
     ```bash
     blender --background --python seu_script.py
     ```
   - Nesse modo, `bpy`, `mathutils` e `BVHTree` já estão disponíveis.

2. **Usar `bpy` como módulo Python**
   - Instale uma build de `bpy` compatível com sua versão de Blender/Python.
   - Depois execute seus scripts normalmente em `python`.
   - Observação: compatibilidade de versão é rígida (ABI).

### O que instalar para o backend Cython

Se quiser backend compilado paralelo:

```bash
python -m pip install cython setuptools wheel
cd cython
python setup.py build_ext --inplace
```

Depois copie o binário `meshcut_parallel*.so`/`.pyd` para a raiz do addon.
Sem esse binário, o addon usa backend Python (BVH) automaticamente.


## Modo produção (Cython obrigatório)

Para uso direto na UI com máximo desempenho, habilite em `Mesh Cut > Settings`:

- `Require Cython Backend` = ON
- `Performance Guard` = ON
- `Always Finish Export` = ON

Com isso, o addon exige o módulo compilado `meshcut_parallel` para exportar.
Se o módulo não estiver presente, a exportação é bloqueada com mensagem de build.

## Garantia de finalização (DXF/SVG)

Quando `Always Finish Export` estiver ativo e o limite de rays for atingido,
o addon degrada para visibilidade adaptativa no restante da cena em vez de cancelar o arquivo.

Resultado: evita travamento e mantém entrega do export final.
