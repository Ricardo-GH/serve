# TorchServe



TorchServe 可将PyTorch模型快速简单地挂载为服务端

**如需阅读全部文档，请查看[PyTorch模型服务端文档](#)**
## TorchServe 结构



![TorchServe 结构](https://user-images.githubusercontent.com/880376/83180095-c44cc600-a0d7-11ea-97c1-23abb4cdbe4d.jpg)
### 术语解释：
* **Frontend:** 这是TorchServe request/response(请求/响应) 处理组件。Frontend用以处理来自客户端的request/response请求，管理Model的生命周期。
* **Model Workers:** Workers负责在模型上运行实时推理，它们是实时运行的模型实例。
* **Model:** Model 可以是`script_module`(使用[TORCH.JIT.SAVE()](https://pytorch.org/docs/stable/generated/torch.jit.save.html)保存)也可以是`eager_mode_models`。Model可以使用自定义的预处理或后处理方法，也可以使用例如`state_dicts`等Model参数。Model可以从云端存储或本地进行加载。
* **Plugins:** Plugins 可以是在TorchServe启动时进行加载的endpoints、authz/authn(认证类/授权类)或是批处理算法。
* **Model Store:** 这是所有可加载模型的存放目录。
## 文档目录


* [TorchServe安装](#torchserve安装)
* [Windows环境下安装TorchServe]()
* [WSL(Windows Subsystem for Linux)环境下安装TorchServe]()
* [装载模型](#装载模型)
* [使用docker快速开始](#使用docker快速开始)
* [开源贡献](#开源贡献)

## TorchServe安装

1. 安装Java 11

   Ubuntu环境:

   ```bash
   sudo apt-get install openjdk-11-jdk
   ```

   MacOS环境: 

   ```bash
   brew tap AdoptOpenJDK/openjdk
   brew cask install adoptopenjdk11
   ```

2. 安装Python依赖包

    CPU或Cuda 10.2环境：

    ```bash
    pip install -U -r requirements.txt
    ```

    Cuda 10.1环境：

    ```bash
    pip install -U -r requirements_cu101.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

3. 安装`torchseve`和`torch-model-archiver`

    [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)环境：

    ```bash
    conda install torchserve torch-model-archiver -c pytorch
    ```

    Pip环境：

    ```bash
    pip install torchserve torch-model-archiver
    ```

  **注意**：Conda 环境下需要 python3.8 来运行`TorchServe`

  现在你已经可以[使用TorchServe封装和装载模型]()了

  ### 安装TorchServe用作开发

  如需使用 TorchServe 进行开发并且想更改其中的源代码，你应该使用源代码方式进行安装。

  如果你在 Conda 环境中，请停用并退出，随后在源代码根目录执行脚本。

  **注意**：执行这个脚本将会卸载已经安装的`torchserve`和`torch-model-archiver`

  #### 基于Debian的系统或MacOS：

  ```
  python ./scripts/install_from_src.py
  ```

  如需查看有关 model archiver 的更多信息，请参阅[细节文档]()

## 装载模型

  这一节将给您展示一个使用 TorchServ e装载模型的样例。如需完整运行样例，你应该完成[TorchServe和model archiver安装]()。

  运行样例，首先 clone TorchServe 仓库

  ```bash
  git clone https://github.com/pytorch/serve.git
  ```

  接下来将在仓库根目录的父级目录进行操作。如果你将仓库 clone 到`/home/my_path/serve`，你需要在`/home/my_path`中进行后续操作。
  
  ### 存储模型

  在使用 TorchServe 装载模型前，需要将模型封装成一个MAR文件。你可以使用 model archiver 来封装模型。你也可以自己创建模型存储目录来存放自己封装好的模型。

  1. 创建模型存储目录

     ```bash
     mkdir model_store
     ```
   

  2. 下载预训练模型

     ```bash
     wget https://download.pytorch.org/models/densenet161-8d451a50.pth
     ```

  3. 使用 model archiver 来封装模型。`extra-files`变量是`TorchServe`默认的`index_to_name`文件，如有需要请更换路径。

     ```bash
     torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --serialized-file\
     densenet161-8d451a50.pth --export-path model_store --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier
     ```

     如需查看有关`torch-model-archiver`的更多信息，请参阅[细节文档]()。

  ### 启动TorchServe服务并装载模型

  当你完成模型的封装与存储之后，使用`torchserve`命令来装载模型。

  ```bash
  torchserve --start --ncs --model-store model_store --models densenet161.mar
  ```

  当你你执行上述`torchserve`命令之后，TorchServe 将在本地启动服务并监听请求。

  **注意：** 如果你在启动 TorchServe 时指定了模型，TorchServe 会自动将可用的 vCPU 数量( CPU 环境下)或可用的 GPU 数量( GPU 环境下)设置成后端 workers。如果是在计算性能很强的主机上运行(拥有大量vCPU 或GPU)，启动过程和 workers 设置过程将会花费一些时间。如果你想减少 TorchServe 的启动时间，应避免在启动时注册和缩放模型，你可以使用相应的 [Management API]() 来将注册和缩放过程延后，[Management API]() 也可以对已分配的特定模型进行更细粒度的控制。

  ### 使用模型进行预测

  向服务器的`predictions`API发送一个请求来进行模型服务器的测试。

  步骤如下：

  - 打开一个新终端（不要在运行TorchServe的终端中进行测试）
  - 使用`curl`来下载一张[猫猫图片](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) 使用`-o`将图片命名为`kitten.jpg`
  - 将猫猫图片使用`curl`向 TorchServe predict 接口发送`POST`请求

  ![猫猫图片](https://github.com/Ricardo-GH/serve/raw/master/docs/images/kitten_small.jpg)

  上述步骤代码如下：

  ```bash
  curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
  curl http://127.0.0.1:8080/predictions/densenet161 -T kitten_small.jpg
  ```

  `predict`接口将预测结果以 JSON 形式返回。返回结果将和如下类似：

  ```json
  [
    {
      "tiger_cat": 0.46933549642562866
    },
    {
      "tabby": 0.4633878469467163
    },
    {
      "Egyptian_cat": 0.06456148624420166
    },
    {
      "lynx": 0.0012828214094042778
    },
    {
      "plastic_bag": 0.00023323034110944718
    }
  ] 
  ```

  你可以在`curl`请求`predict`接口的响应、服务器的`log`和运行 TorchServe 的终端中看到结果。结果也会被 [Metrics]() 记录在服务器。

  以上就是使用 TorchServe 装载深度学习模型的过程。[了解更多]()

  ### 停止 TorchServe 服务

  如需停止正在运行的 TorchServe 实例，执行以下命令：

  ```bash
  torchserve --stop
  ```

  你将会在终端看到 TorchServe 已经停止的输出。

  ### 并发数和线程数设置

  TorchServe 提供了可供用户在 CPU 或 GPU 环境下配置线程数的接口。~~根据工作量的不同，这是一个非常重要的配置，加快服务器启动~~。*注意：以下属性将对高负载任务产生较大影响。* 如果 TorchServe 运行在多 GPU 环境下，可以通过`number_of_cpu`属性来设置服务器挂载的每个模型使用 GPU 的数量。如果我们在服务器上挂载的多个模型，这将会应用到所有的模型上。如果这个数值设置的较低(比如0或1)，将会导致 GPU 利用率偏低。相对的，如果这个数值设置的过高(≥系统可用的GPU数量)，将每个模型上生成对应数量的 workers ，显然这会对 GPU 资源产生不必要的争夺，最终会导致线程到 GPU 的调度降低。

  ```shell 
  number_of_cpu = (实机 GPU 数量) / (不同模型数量)
  ```
  ## 使用 docker 快速开始
  具体请参考[TorchServe Docker](#)

  ## 了解更多
  * [TorchServe 全部文档](#)
  * [管理模型 API](#)
  * [封装模型以在 TorchServe 上使用](#)
  * [TorchServe 自有的预训练及预封装模型](#)
  
  ## 开源贡献

  我们欢迎所有的开源贡献！

  想要了解如何进行开源贡献，可以参考[这里](#)的贡献者指南。

  如需提交 bug 或 feature ，请在 GitHub 上提交 issue 。如需提交 Pull Request，请参考[模板]()。Cheers~

  ## 免责声明
  这个仓库由 Amazon，Facebook 以及众多个人开发者（详见[个人开发者名单]()）共同开发并维护。如有关于 Facebook 的问题，请发送邮件至opensource@fb.com。如有关于 Amazon 的问题，请发送邮件至torchserve@amazon.com。如有其他问题，请参看本仓库的[issue]()

  **TorchServe 项目由 [Mutil Model Server(MMS)](https://github.com/awslabs/multi-model-server) 发展而来**

  
  