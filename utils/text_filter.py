import re
import pdb

# Add rules as patterns.
# remove : remove matched substring from string
# change : change matched substring as substute resource
#
#
# Order of adding rule -> Order of filtering
# To filter with rules, call filter() method


class TextFilter:

  def __init__(self):
    self.rules = []
    self.rules_for_detect = []

  # Private inner classes
  class Remove_rule:
    def __init__(self, pattern):
      self.pattern = re.compile(pattern)

    def filter(self, sentence):
      return self.pattern.sub('', sentence)

  class Replace_rule:
    def __init__(self, pattern, substitute, flags=None, unicode=False):
      if unicode:
        self.pattern = re.compile(pattern, re.UNICODE)
      else:
        self.pattern = re.compile(pattern)
      self.substitute = substitute
      self.flags = flags

    def filter(self, sentence):
      if self.flags is not None:
        return self.pattern.sub(self.substitute, sentence, self.flags)
      else:

        # print(sentence)
        # if 'include' in sentence:
        #   pdb.set_trace()

        return self.pattern.sub(self.substitute, sentence)

  class Detection_rule:
    def __init__(self, detect_field, pattern):
      self.detect_field = detect_field
      self.pattern = re.compile(pattern)


    def detect(self, sentence):
      out = self.pattern.search(sentence)
      data = dict()
      if out is not None:
        data[self.detect_field] = True
        return data
      data[self.detect_field] = False
      return data


  # regex sub, string replace, string pos 잡고 컷, 리무브...

  def filter(self, sentence):
    for rule in self.rules:
      sentence = rule.filter(sentence)

    sentence = sentence.strip()

    return sentence

  def detect(self, sentence):
    check_dict = dict()
    for rule in self.rules_for_detect:
      out = rule.detect(sentence)
      check_dict.update(out)
    return check_dict



  """
  위키피디아 전처리를 위한 Replace/ Remove Rule (doc open, close)
  """

  def detect_doc_open_pattern(self):
    pattern = r"^<doc.*>$"
    self.rules_for_detect.append(self.Detection_rule('doc_open', pattern))

  def detect_doc_close_pattern(self):
    pattern = r"</doc>"
    self.rules_for_detect.append(self.Detection_rule('doc_close', pattern))




  def change_match_pattern_to_subs(self, pattern, substitute):
    self.rules.append(self.Replace_rule(pattern, substitute))

  def remove_match_pattern(self, pattern):
    self.rules.append(self.Remove_rule(pattern))

  def remove_from_pattern_to_space(self, pattern):
    pattern = pattern + r'.*?($|[ \t\n\r])'
    self.rules.append(self.Remove_rule(pattern))

  def remove_from_space_to_pattern(self, pattern):
    pass

  def remove_from_pattern_to_pattern(self, pattern_start, pattern_end):
    pattern = pattern_start + r'.*?' + pattern_end
    self.rules.append(self.Remove_rule(pattern))

  def remove_http_url(self):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&#+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    self.rules.append(self.Remove_rule(pattern))

  def remove_ftp_url(self):
    pattern = r'ftp://(?:[a-zA-Z]|[0-9]|[$-_@.&#+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    self.rules.append(self.Remove_rule(pattern))

  def remove_unstructured_url(self):
    pass

  def remove_emails(self):
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    self.rules.append(self.Remove_rule(pattern))

  def remove_phone_number(self):
    pattern = r'[0-9]+(-[0-9]+)+'
    self.rules.append(self.Remove_rule(pattern))

  # 한글, 한글자모, 영어대소문자, 숫자, 키보드 내 특수문자 제외하고 모두 삭제
  def remove_trash_char(self):
    pattern = r"[^가-힣a-zA-Z0-9ㄱ-ㅣ\`\~\!\@\#\$\%\^\&\*\(\)\-\_\=\+\|\\\[\]\{\}\;\:\'\"\,\.\<\>\?\/\t ]+"
    self.rules.append(self.Remove_rule(pattern))

  # handle wiki
  def remove_doc_open_tag(self):
    pattern = r"^<doc.*>$"
    self.rules.append(self.Remove_rule(pattern))

  # handle wiki
  def remove_doc_close_tag(self):
    pattern = r"</doc>"
    self.rules.append(self.Remove_rule(pattern))





  """
  나무위키 전처리를 위한 Replace/ Remove Rule
  """

  # handle namu-wiki
  def _replace_chinese(self):
    pattern = u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]'
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute, flags, unicode=True))

  # handle namu-wiki
  def _replace_japanese(self):
    pattern = u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]'
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute, flags, unicode=True))

  # remove html
  def _replace_html(self):
    pattern = r"\{\{\{#\!html[^\}]*\}\}\}"
    substitute = ''
    flags=re.IGNORECASE|re.MULTILINE|re.DOTALL
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove redirect
  def _replace_redirect(self):
    pattern = r"#redirect .*"
    substitute = ''
    flags=re.IGNORECASE
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

    # remove 분류

  def _replace_tag0(self):
    pattern = r"틀:.*"
    substitute = ''
    flags = None
    self.rules.append(self.Replace_rule(pattern, substitute, flags=flags))

  # remove 분류
  def _replace_tag1(self):
    pattern = r"\[\[분류:.*"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove 파일
  def _replace_tag2(self):
    pattern = r"\[\[파일:.*"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove 상위문서
  def _replace_tag3(self):
    pattern = r"\* 상위 문서 ?:.*"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove youtube
  def _replace_tag4(self):
    pattern = r"\[youtube\(\w+\)\]"
    substitute = ''
    flags=re.IGNORECASE
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove include
  def _replace_tag5(self):
    pattern = r"\[(i|I)nclude\(([^\]|]*)(\|[^]]*)?\]"
    substitute =  r'\1'
    flags=re.IGNORECASE
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove link
  def _replace_tag6(self):
    pattern = r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]"
    substitute = r'\1'
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove 각주
  def _replace_tag7(self):
    # pattern = r"\[([^\]]*)\]"
    pattern = r"\[.*\]"
    substitute = ''
    flags = None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))


  # remove text color/size
  def _replace_tag8(self):
    pattern = r"\{\{\{([^\ }|]*) ([^\}|]*)\}\}\}"
    substitute =  r'\2'
    flags = None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove text bold
  def _replace_tag9(self):
    pattern = r"'''([^']*)'''"
    substitute = r'\1'
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove strike-through
  def _replace_tag10(self):
    pattern = r"(~~|--)([^']*)(~~|--)"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # remove table
  def _replace_tag11(self):
    pattern = r"\|[\S\s]*\|"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # 문단 제거
  def _replace_tag12(self):
    pattern = r"=(.*)="
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # 동영상 제거
  def _replace_tag13(self):
    pattern = r"width.*\/iframe>"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # html tag
  def _replace_tag14(self):
    pattern = r"<.*?>"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # html tag
  def _replace_tag15(self):
    pattern = r"\{\{\|([^|]*)\|\}\}"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # 수식 표현 제거
  def _replace_tag16(self):
    pattern = r"\\[^가-힣]*"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # 괄호 내부 내용 제거
  def _replace_tag17(self):
    pattern = r"\([^)]*\)"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))

  # include tag 제거
  def _replace_tag18(self):
    pattern = r"\[(i|I)nclude.*\]"
    substitute = ''
    flags=None
    self.rules.append(self.Replace_rule(pattern, substitute,flags=flags))


  def replace_namu_tags(self):
    self._replace_tag18()
    self._replace_chinese()
    self._replace_japanese()
    self._replace_html()
    self._replace_redirect()
    self._replace_tag0()
    self._replace_tag1()
    self._replace_tag2()
    self._replace_tag3()
    self._replace_tag4()
    self._replace_tag5()
    self._replace_tag6()
    self._replace_tag7()
    self._replace_tag8()
    self._replace_tag9()
    self._replace_tag10()
    self._replace_tag11()
    self._replace_tag12()
    self._replace_tag13()
    self._replace_tag14()
    self._replace_tag15()
    self._replace_tag16()
    self._replace_tag17()


  def spacing_on_both_side_of_mark(self):
    pattern = r'[\`\~\!\@\#\$\%\^\&\*\(\)\-\_\=\+\[\{\]\}\\\|\;\:\'\"\,\.\/\<\>\?]'
    substitute = r' \g<0> '
    self.rules.append(self.Replace_rule(pattern, substitute))

  def spacing_on_both_side_of_hangul_jamo(self):
    pattern = r'[ㄱ-ㅣ]+'
    substitute = r' \g<0> '
    self.rules.append(self.Replace_rule(pattern, substitute))

  # 문장 내부에 공백(' ', \n \t \r)이 여러번 있는 경우 space 하나로 치환
  def strip_inside(self):
    pattern = r'[ \t\n\r]+'
    substitute = ' '
    self.rules.append(self.Replace_rule(pattern, substitute))






