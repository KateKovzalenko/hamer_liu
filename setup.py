from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='hamer',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'xtcocotools==1.14.3',
        'gdown==5.2.0',
        'opencv-python==4.11.0.86',
        'pyrender==0.1.45',
        'pytorch-lightning==2.5.5',
        'scikit-image==0.25.2',
        'smplx==0.1.28',
        'torch==2.2.0+cu118',
        'torchvision==0.17.0+cu118',
        'yacs==0.1.8',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        'chumpy  @ git+https://github.com/mattloper/chumpy',
        'mmcv==1.3.9',
        'timm==1.0.20',
        'einops==0.8.1',
        'pandas==2.3.3',
    ],
    extras_require={
        'all': [
            'hydra-core==1.3.2',
            'hydra-submitit-launcher==1.2.0',
            'hydra-colorlog==1.2.0',
            'pyrootutils==1.0.4',
            'rich==14.1.0',
            'webdataset==1.0.2',
        ],
    },
)
