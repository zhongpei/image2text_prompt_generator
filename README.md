# Image2text Prompt Generator

图片 Prompt 解析工具。这是一款方便快捷的工具，支持一键从图片中解析出 Prompt 描述，并能够基于描述进行扩展，以便进行二次图片生成。

我们的工具支持直接使用中文进行原始 Prompt 描述，这使得使用者可以非常便捷地输入他们的想法，而不用担心语言障碍。同时，我们的工具还支持将中文转换为模型生成效果更好的英文 Prompt 描述。这使得生成的图片更加准确、真实。

使用我们的工具，您可以在短短几分钟内轻松地生成您想要的图片，而无需具备深度学习的专业知识。这款工具是极其易用的，只需上传一张图片，输入一些描述，并按下“生成”按钮，即可获得您想要的图片。

总之，我们的工具是一款非常强大的图片 Prompt 解析工具，能够帮助您快速生成高质量的图片，而不需要任何专业知识。如果您正在寻找一款能够方便快捷地生成图片的工具，那么不妨试试我们的产品吧！

我们的产品支持多种深度学习模型，包括 CLIP（Contrastive Language-Image Pre-Training）、BLIP（Bridging Language and Image Pre-Training）和 WD14（Wide and Deep 14-layer）等。

CLIP 模型是一种强大的自然语言处理和计算机视觉模型，它能够将文本描述和图像特征联系起来，从而更准确地理解用户的需求并生成所需的图像。BLIP 模型则是一个跨模态的预训练模型，它结合了语言和图像信息，可以用于生成图像、图像搜索和视觉问答等任务。WD14 模型则是一种深度神经网络模型，用于处理结构化和非结构化数据的大规模机器学习任务。

# 一键包

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


