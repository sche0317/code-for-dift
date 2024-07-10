from ml_collections import ConfigDict
from ml_collections import config_dict

def vae_config():
    config = config_dict.ConfigDict()
    config.encoder = {
                    'in_channels': 3,
                    'out_channels': 4,
                    'block_out_channels': (128, 256, 512, 512),
                    'layers_per_block': 2,
                    'norm_num_groups': 32,
                    'double_z':  True
    }
    
    config.decoder = {
                    'in_channels': 4,
                    'out_channels': 3,
                    'block_out_channels': (128, 256,512,512),
                    'layers_per_block': 2,
                    'norm_num_groups': 32,
                    'output_size': 256
    }
    return config

def med_vae_config():
    config = config_dict.ConfigDict()
    config.encoder = {
                    'in_channels': 1,
                    'out_channels': 2,
                    'block_out_channels': (64, 128, 256, 256),
                    'layers_per_block': 2,
                    'norm_num_groups': 16,
                    'double_z':  True
    }
    
    config.decoder = {
                    'in_channels': 2,
                    'out_channels': 1,
                    'block_out_channels': (64, 128, 256, 256),
                    'layers_per_block': 2,
                    'norm_num_groups': 16,
                    'output_size': 224
    }
    return config