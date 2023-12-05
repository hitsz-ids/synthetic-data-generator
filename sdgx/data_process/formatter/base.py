# Formatter： 列格式转换工具，基本描述如下：
#   - 针对不同类型的列，实现解析能力，例如：DataTime 搞成时间戳形式；
#   - 针对不同类型的列，提供格式上的转换能力
#   - 输入和输出均为【列】数据
#
# 同时也在此说明与 transform 的区别：
#   - 涉及到【单列】作为输入的，涉及【格式转换】问题，使用 formatter
#   - 涉及到【整张表】作为输入的进行转换，使用 data transformer
#   - 通常，在 Data Transformer 的实现中，针对列的情况，调用不同的 formatter
#   - 提供 extract 方法
#


class BaseFormatter(object):
    # def extract_xxx(self):
    #     pass

    pass
