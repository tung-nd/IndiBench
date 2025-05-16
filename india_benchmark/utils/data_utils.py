CONSTANTS = [
    'MTERH',
    'LAND'
]

SUPPORT_VARIABLES = [
    'MTERH',
    'TMP',
    'UGRD',
    'VGRD',
    'PRMSL',
    'TCDCRO',
    'APCP',
    'LAND',
    'HGT',
    'UGRD_prl',
    'VGRD_prl',
    'TMP_prl',
    'RH'
]

PRESSURE_VARIABLES = ['HGT', 'UGRD_prl', 'VGRD_prl', 'TMP_prl', 'RH']
PRESSURE_LEVELS = [925, 850, 700, 600, 500, 250, 50]
SURFACE_VARIABLES = [v for v in SUPPORT_VARIABLES if v not in PRESSURE_VARIABLES]

IMDAA_TO_ERA5_MAPPING = {
    'MTERH': 'terrain_height',
    'LAND': 'land_cover',
    'TMP': '2m_temperature',
    'UGRD': '10m_u_component_of_wind',
    'VGRD': '10m_v_component_of_wind',
    'APCP': 'total_precipitation',
    'PRMSL': 'mean_sea_level_pressure',
    'TCDCRO': 'total_cloud_cover',
    'HGT': 'geopotential',
    'UGRD_prl': 'u_component_of_wind',
    'VGRD_prl': 'v_component_of_wind',
    'TMP_prl': 'temperature',
    'RH': 'relative_humidity',
}

for variable in PRESSURE_VARIABLES:
    for level in PRESSURE_LEVELS:
        IMDAA_TO_ERA5_MAPPING[f'{variable}{level}'] = f'{IMDAA_TO_ERA5_MAPPING[variable]}_{level}'
    IMDAA_TO_ERA5_MAPPING.pop(variable)