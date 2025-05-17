from PyInstaller.utils.hooks import collect_data_files

# Add config.yaml
datas = [
    ('config/config.yaml', 'config'),
    ('models/*.py', 'models'),
    ('utils/*.py', 'utils')
]

# Collect all needed packages
hiddenimports = [
    'models.detector',
    'models.metrics',
    'utils.data_loader',
    'utils.descriptor',
    'train',
    'inference'
]