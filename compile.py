import compileall

if __name__=="__main__":
    # 指定要编译的目录
    directory_path = '/home/yingmuzhi/SpecML2/data/compile'

    # 将指定目录下的所有 Python 文件编译成 .pyc 文件
    compileall.compile_dir(directory_path)
