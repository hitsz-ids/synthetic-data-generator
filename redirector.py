import sys
import io
import time


class WriteableRedirector(io.TextIOBase):
    def __init__(self, path=f"./log_{time.time()}.txt"):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, 'w')
        print(f"重定向开启 > {self.path}")
        sys.stdout = self
        sys.stderr = self
        self.content = ""
        return self

    def write(self, text):
        self.f.write(text)
        sys.__stdout__.write(text)
        self.content += text

    def clear(self):
        self.content = ""

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__
        print("重定向关闭")
