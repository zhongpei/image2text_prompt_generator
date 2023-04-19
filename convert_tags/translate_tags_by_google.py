import googletrans
from googletrans import Translator

print(googletrans.LANGUAGES)

translator = Translator()
result = translator.translate('hello world','zh-cn')
print(result.text)
lines = []
with open("all.txt","r") as f:
    lines = f.readlines()
with open("all_zh.text","a+") as f:
    for l in lines:
        l = l.replace("_"," ").strip()
        result = translator.translate(l,'zh-cn')
        print("{}={}".format(l,result.text))
        f.write("{}={}".format(l,result))
