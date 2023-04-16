# Image2text Prompt Generator

图片 Prompt 解析工具。这是一款方便快捷的工具，支持一键从图片中解析出 Prompt 描述，并能够基于描述进行扩展，以便进行二次图片生成。

* 在线演示 [demo](https://huggingface.co/spaces/hahahafofo/image2text_prompt_generator)
* 图片功能请本地部署，使用本地GPU
* 部分模型使用CPU，防止GPU显存溢出
* 支持stable diffusion和midjourney两种prompt生成方式
* 支持chatglm描述画面生成

支持多种深度学习模型，包括 CLIP（Contrastive Language-Image Pre-Training）、BLIP（Bridging Language and Image Pre-Training）和 WD14（Wide and Deep 14-layer）等。

多种模型支持
![img.png](./img/param.png)

## 文生文
![img.png](./img/text2text.png)

## 图生文
![img.png](./img/image2text.png)

## chatglm 生成
![img.png](./img/chatglm.png)

## 一键包

[baidu云盘](https://pan.baidu.com/s/1pKtpPmiuliX7rf0z-5HY_w?pwd=79sk) 提取码: 79sk

解压缩后，点击webui.bat
第一次运行会自动下载模型

# 安装工具的说明如下：

首先，确保您的计算机已经安装了 Python 3。如果您尚未安装 Python，请前往官方网站（https://www.python.org/downloads/）下载并安装最新版本的 Python 3。
接着，下载并解压缩我们的工具安装包。
打开命令行窗口（Windows 用户可以按下 Win + R 键，在运行框中输入 “cmd” 并按下回车键打开命令行窗口），并进入到工具安装包所在的目录。
在命令行窗口中输入以下命令安装所需的依赖项：


```bash
git clone https://huggingface.co/spaces/hahahafofo/image2text_prompt_generator
cd image2text_prompt_generator

# 建立虚拟环境
python -m "venv" venv
# 激活环境 linux & mac 
./venv/bin/activate
# 激活环境 windows
.\venv\Scripts\activate


# gpu version
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install --upgrade -r requirements.txt
  
```

这将自动安装所需的 Python 依赖项。
安装完成后，您可以运行以下命令启动工具：
```bash
# 激活环境 linux & mac
./venv/bin/activate
# 激活环境 windows
.\venv\Scripts\activate

# 运行程序
python app.py
    
```


这将启动工具并在您的浏览器中打开工具的主页。如果您的浏览器没有自动打开，请手动输入以下网址：http://localhost:7860/
工具现在已经成功安装并启动了。您可以按照工具的说明文档，开始使用它来处理您的图片数据。


