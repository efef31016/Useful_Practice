from setuptools import setup, find_packages

setup(
    name='ChatBotModel',              # 套件名稱
    version='1.0.0',                  # 版本號
    author='Jiao',               # 作者名稱
    author_email='yhocotw31016@gmail.com',   # 作者電子郵件
    description='We can chat with AI BOT',  # 套件描述
    packages=find_packages(),         # 自動尋找所有套件資料夾
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)