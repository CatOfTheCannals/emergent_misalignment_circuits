from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
req_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if req_path.exists():
    with req_path.open() as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="emergent_misalignment_circuits",
    version="0.1.0",
    description="Pipeline for exploring pivotal-token misalignment circuits in language models",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
) 