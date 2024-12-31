from setuptools import setup, find_packages

setup(
    name='zero',       # 包名
    version='0.0.1',   # 版本号
    author='xqli',     # 作者
    author_email='xqli@ubiquant.com',                # 作者邮箱
    description='A Quant Tech Simulator Framework',  # 项目简介
    long_description=open('README.md').read(),       # 长描述，通常是README文件
    long_description_content_type='text/markdown',   # 长描述的格式
    url='https://github.com/CVPaul/zero',            # 项目URL
    packages=find_packages(),  # 自动查找包
    install_requires=[  # 项目依赖
        # 'dependency1>=1.0.0',
        # 'dependency2==1.2.3',
    ],
    classifiers=[  # 项目分类
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # 兼容的Python版本
)
