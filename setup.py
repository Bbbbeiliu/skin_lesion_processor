
from setuptools import setup, find_packages

setup(
    name="skin-lesion-processor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "geomdl>=5.3.0",
        "ezdxf>=0.17.0",
    ],
    entry_points={
        'console_scripts': [
            'skin-lesion-processor=main:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Skin Lesion Contour Processing System with NURBS fitting",
    keywords="skin lesion, contour, NURBS, DXF, medical imaging",
    python_requires=">=3.8",
)