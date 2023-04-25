import re
from typing import List

from langchain.text_splitter import CharacterTextSplitter


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, min_length=200, max_length=512, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.min_length = min_length
        self.max_length = max_length
        self.sent_sep_pattern = re.compile('([﹒﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；

    def fix_length(self, sent_list: List[str]) -> List[str]:
        new_sent_list = []

        for i in range(len(sent_list)):
            last_sent = new_sent_list[-1] if new_sent_list else ""
            sent = sent_list[i]

            if len(last_sent) < self.min_length and new_sent_list:
                new_sent_list[-1] += sent
            elif len(sent) > self.max_length:
                new_sent_list.extend(self.__split_text(sent, re.compile('([,，;；])')))
            else:
                new_sent_list.append(sent)
        return new_sent_list

    def split_text(self, text: str) -> List[str]:
        sent_list = self.__split_text(text)
        print(f"split_text: {len(sent_list)}")
        sent_list = self.fix_length(sent_list)
        print(f"fix_length: {len(sent_list)}")

        return sent_list

    def __split_text(self, text: str, sent_sep_pattern=None) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")

        if sent_sep_pattern is None:
            sent_sep_pattern = self.sent_sep_pattern

        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


if __name__ == '__main__':
    test_text = "我是中国人,我爱我的祖国。"
    text = "".join([test_text] * 100)
    splitter = ChineseTextSplitter(min_length=22, max_length=50)

    print([(l, len(l)) for l in splitter.split_text(text)])
    splitter = ChineseTextSplitter(min_length=10, max_length=20)
    print([(l, len(l)) for l in splitter.split_text(text)])
