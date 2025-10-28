from mjlab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from mjlab.terrains import (
    BoxRandomGridTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    BoxFlatTerrainCfg,
)

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "boxes": BoxRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": BoxFlatTerrainCfg(
            proportion=0.1
        ),
    }
)

TERRAINS_IMPORTER_CFG = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=TERRAINS_CFG,
    max_init_terrain_level=5,
    env_spacing=2.5,
)
