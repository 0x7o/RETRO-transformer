import setuptools

setuptools.setup(
	name="retro_transformer",
	version="1.0.1",
	author="Danil Kononyuk",
	author_email="me@0x7o.link",
	description="Easy-to-use Retrieval-Enhanced Transformer (Retro) implementation",
	long_description="""# Retrieval-Enhanced Transformer

Easy-to-use [Retro](https://arxiv.org/abs/2112.04426) implementation in PyTorch.""",
	long_description_content_type="text/markdown",
	url="https://github.com/0x7o/RETRO-transformer",
	packages=setuptools.find_packages(),
	classifiers=[],
	python_requires='>=3.6',
)