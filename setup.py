import setuptools


setuptools.setup(
    name="anmi",  # Replace with your own username
    version="0.0.6",
    author="alexmascension",
    author_email="alexmascension@gmail.com",
    description="Archivos para correr ANMI",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/alexmascension/ANMI",
    project_urls={"Bug Tracker": "https://github.com/alexmascension/ANMI/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
