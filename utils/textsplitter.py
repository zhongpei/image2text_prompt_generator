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
        self.sent_sep_pattern_2 = re.compile('([,，;；])')

    def fix_length(self, sent_list: List[str]) -> List[str]:
        new_sent_list = []

        for i in range(len(sent_list)):
            last_sent = new_sent_list[-1] if new_sent_list else ""
            sent = sent_list[i]

            if len(last_sent) < self.min_length and new_sent_list:
                new_sent_list[-1] += sent
            elif len(sent) > self.max_length:
                new_sent_list.extend(self.__split_text(sent, self.sent_sep_pattern_2))
            else:
                new_sent_list.append(sent)
        return new_sent_list

    def split_text(self, text: str) -> List[str]:
        sent_list = self.__split_text(text)

        fix_list = self.fix_length(sent_list)
        print(f"text: {len(text)} ==> split: {len(sent_list)}   fix: {len(fix_list)}")

        return fix_list

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
    test_text = """娘儿们说了回话，不觉五更，鸡叫天明。吴月娘梳洗面貌，走到禅堂中，礼佛烧香。只见普静老师在禅床上高叫：“那吴氏娘子，你如何可省悟得了么？”
 
这月娘便跪下参拜：“上告尊师，弟子吴氏，肉眼凡胎，不知师父是一尊古佛。适间一梦中都已省悟了。”
 
老师道：“既已省悟，也不消前去，你就去，也无过只是如此。倒没的丧了五口儿性命。你这儿子，有分有缘遇着我，都是你平日一点善根所种。不然，定然难免骨肉分离。当初，你去世夫主西门庆造恶非善，此子转身托化你家，本要荡散其财本，倾覆其产业，临死还当身首羿处。今我度脱了他去，做了徒弟，常言‘一子出家，九祖升天’，你那夫主冤愆解释，亦得超生去了。你不信，跟我来，与你看一看。”
 
于是叉步来到方丈内，只见孝哥儿还睡在床上。老师将手中禅杖，向他头上只一点，教月娘众人看。忽然翻过身来，却是西门庆，项带沉枷，腰系铁索。复用禅杖只一点，依旧是孝哥儿睡在床上。月娘见了，不觉放声大哭，原来孝哥儿即是西门庆托生。
 
良久，孝哥儿醒了。月娘问他：“如何你跟了师父出家。"""
    text = "".join([test_text] * 1)
    splitter = ChineseTextSplitter(min_length=100, max_length=500, pdf=False)

    print([(l, len(l)) for l in splitter.split_text(text)])
    splitter = ChineseTextSplitter(min_length=10, max_length=20)
    print([(l, len(l)) for l in splitter.split_text(text)])
