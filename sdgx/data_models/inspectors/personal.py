import re

from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.inspectors.regex import RegexInspector

chn_last_names = r"[赵|钱|孙|李|周|吴|郑|王|冯|陈|褚|卫|蒋|沈|韩|杨|朱|秦|尤|许|何|吕|施|张|孔|曹|严|华|金|魏|陶|姜|戚|谢|邹|喻|柏|水|窦|章|云|苏|潘|葛|奚|范|彭|郎|鲁|韦|昌|马|苗|凤|花|方|俞|任|袁|柳|酆|鲍|史|唐|费|廉|岑|薛|雷|贺|倪|汤|滕|殷|罗|毕|郝|邬|安|常|乐|于|时|傅|皮|卞|齐|康|伍|余|元|卜|顾|孟|平|黄|和|穆|萧|尹|姚|邵|湛|汪|祁|毛|禹|狄|米|贝|明|臧|计|伏|成|戴|宋|茅|庞|熊|纪|舒|屈|项|祝|董|梁|杜|阮|蓝|闵|席|季|麻|贾|路|娄|江|童|颜|郭|梅|盛|林|刁|徐|邱|骆|夏|蔡|田|樊|胡|凌|霍|虞|万|支|柯|昝|管|卢|莫|经|房|裘|缪|干|解|应|宗|丁|宣|贲|邓|郁|杭|洪|包|诸|左|石|崔|吉|钮|龚|程|嵇|邢|滑|裴|陆|荣|翁|荀|於|惠|甄|麴|家|封|芮|羿|储|靳|汲|邴|糜|松|井|段|富|巫|乌|焦|巴|弓|牧|隗|山|谷|车|侯|宓|蓬|全|郗|班|仰|仲|伊|宫|宁|仇|栾|暴|甘|钭|历|戎|祖|武|符|刘|景|詹|束|龙|叶|幸|司|韶|郜|黎|溥|印|宿|白|怀|蒲|邰|从|鄂|索|咸|籍|卓|蔺|屠|蒙|池|乔|阳|郁|胥|能|苍|闻|莘|翟|谭|贡|劳|逄|姬|申|扶|堵|冉|宰|郦|雍|却|桑|桂|濮|牛|寿|通|边|扈|燕|冀|浦|尚|农|温|别|庄|晏|柴|瞿|充|慕|连|茹|习|宦|艾|鱼|容|向|古|易|慎|戈|廖|庾|终|暨|居|衡|步|都|耿|满|弘|匡|国|文|寇|广|禄|阙|东|欧|沃|利|蔚|越|夔|隆|师|巩|厍|聂|晁|勾|敖|融|冷|訾|辛|阚|那|简|饶|空|曾|毋|沙|乜|养|鞠|须|丰|巢|关|蒯|相|荆|红|游|竺|权|司马|上官|欧阳|夏侯|诸葛|闻人|东方|赫连|皇甫|尉迟|公羊|澹台|公冶宗政|濮阳|淳于|单于|太叔|申屠|公孙|仲孙|轩辕|令狐|钟离|宇文|长孙|慕容|司徒|司空|召|有|舜|岳|黄辰|寸|贰|皇|侨|彤|竭|端|赫|实|甫|集|象|翠|狂|辟|典|良|函|芒|苦|其|京|中|夕|乌孙|完颜|富察|费莫|蹇|称|诺|来|多|繁|戊|朴|回|毓|鉏|税|荤|靖|绪|愈|硕|牢|买|但|巧|枚|撒|泰|秘|亥|绍|以|壬|森|斋|释|奕|姒|朋|求|羽|用|占|真|穰|翦|闾|漆|贵|代|贯|旁|崇|栋|告|休|褒|谏|锐|皋|闳|在|歧|禾|示|是|委|钊|频|嬴|呼|大|威|昂|律|冒|保|系|抄|定|化|莱|校|么|抗|祢|綦|悟|宏|功|庚|务|敏|捷|拱|兆|丑|丙|畅|苟|随|类|卯|俟|友|答|乙|允|甲|留|尾|佼|玄|乘|裔|延|植|环|矫|赛|昔|侍|度|旷|遇|偶|前|由|咎|塞|敛|受|泷|袭|衅|叔|圣|御|夫|仆|镇|藩|邸|府|掌|首|员|焉|戏|可|智|尔|凭|悉|进|笃|厚|仁|肇|资|仍|九|衷|哀|刑|俎|仵|圭|夷|徭|蛮|汗|孛|乾|帖|罕|洛|淦|洋|邶|郸|郯|邗|邛|剑|虢|隋|蒿|茆|菅|苌|树|桐|锁|钟|机|盘|铎|斛|玉|线|针|箕|庹|绳|磨|蒉|瓮|弭|刀|疏|牵|浑|恽|势|世|仝|同|蚁|止|戢|睢|冼|种|涂|肖|己|泣|潜|卷|脱|谬|蹉|赧|浮|顿|说|次|错|念|夙|斯|完|丹|聊|源|吾|寻|展|不|户|闭|才|愚|霜|烟|寒|少|字|桥|板|斐|独|千|诗|嘉|扬|善|揭|祈|析|赤|紫|青|柔|刚|奇|拜|佛|陀|弥|阿|素|长|僧|隐|仙|隽|宇|祭|酒|淡|塔|琦|闪|始|南|天|接|波|碧|速|禚|腾|潮|镜|似|澄|潭|謇|纵|渠|奈|濯|沐|茂|英|兰|檀|藤|枝|检|生|折|登|驹|骑|貊|虎|肥|鹿|雀|野|禽|飞|节|宜|鲜|粟|栗|豆|帛|官|布|衣|藏|宝|钞|银|门|盈|庆|喜|及|普|建|营|巨|望|希|道|载|声|漫|犁|力|贸|勤|革|改|兴|亓|睦|修|信|闽|守|勇|汉|尉|士|旅|五|令|将|旗|军|行|奉|敬|恭|仪|母|堂|丘|义|礼|慈|孝|理|伦|卿|问|永|辉|位|让|尧|依|犹|介|承|市|所|苑|杞|剧|第|零|谌|招|续|达|忻|六|鄞|战|迟|候|宛|励|粘|萨|邝|覃|辜|初|楼|城|区|局|台|原|考|妫|纳|泉|老|清|德|卑|过|麦|曲|竹|百|福|言|第五|佟|爱|年|笪|谯|哈|墨|连|南宫|赏|伯|佴|佘|牟|商|西门|东门|左丘|梁丘|琴|后|况|亢|缑|帅|微生|羊舌|归|呼延|南门|东郭|百里|钦|鄢|汝|法|闫|楚|晋|谷梁|宰父|夹谷|拓跋|壤驷|乐正|漆雕|公西|巫马|端木|颛孙|子车|督|仉|司寇|亓官|三小|鲜于|锺离|盖|逯|库|郏|逢|阴|薄|厉|稽|闾丘|公良|段干|开|光|操|瑞|眭|泥|运|摩|伟|铁|迮]"


class EmailInspector(RegexInspector):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    data_type_name = "email"

    _inspect_level = 30

    pii = True


class ChinaMainlandIDInspector(RegexInspector):
    pattern = (
        r"^[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$"
    )

    data_type_name = "china_mainland_id"

    _inspect_level = 30

    pii = True


class ChinaMainlandMobilePhoneInspector(RegexInspector):
    pattern = r"^1[3-9]\d{9}$"

    data_type_name = "china_mainland_mobile_phone"

    _inspect_level = 30

    pii = True


# 邮编
class ChinaMainlandPostCode(RegexInspector):
    pattern = r"^[0-9]{6}$"

    _match_percentage = 0.95
    """
    Since zip codes and six-digit integers are the same, here we increase match_percentage to prevent some pure integer columns from being recognized.
    """

    data_type_name = "china_mainland_postcode"

    _inspect_level = 20

    pii = False


# 统一社会信用代码
class ChinaMainlandUnifiedSocialCreditCode(RegexInspector):
    pattern = r"^[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}$"

    data_type_name = "unified_social_credit_code"

    _inspect_level = 30

    pii = True

    pattern_ID = (
        r"^[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$"
    )

    p_id = re.compile(pattern_ID)

    def domain_verification(self, each_sample):
        if re.match(self.p_id, each_sample):
            return False
        return True


class ChinaMainlandAddressInspector(RegexInspector):

    # This regular expression does not take effect and is only for reference by developers.
    # pattern = r"^[\u4e00-\u9fa5]{2,}(省|自治区|特别行政区|市)|[\u4e00-\u9fa5]{2,}(市|区|县|自治州|自治县|县级市|地区|盟|林区)?|[\u4e00-\u9fa5]{0,}(街道|镇|乡)?|[\u4e00-\u9fa5]{0,}(路|街|巷|弄)?|[\u4e00-\u9fa5]{0,}(号|弄)?$"

    pattern = r"^[\u4e00-\u9fa5]{2,}(省|自治区|特别行政区|市|县|村|弄|乡|路|街)"

    pii = True

    data_type_name = "china_mainland_address"

    _inspect_level = 30

    address_min_length = 8

    address_max_length = 30

    def domain_verification(self, each_sample):
        # CHN address should be between 8 - 30 characters
        if len(each_sample) < self.address_min_length:
            return False
        if len(each_sample) > self.address_max_length:
            return False
        # notice to distinguishing from the company name
        if each_sample.endswith("公司"):
            return False
        return True


# 中文姓名
class ChineseNameInspector(RegexInspector):

    pattern = chn_last_names[:600] + r"][\u4e00-\u9fa5]{1,3}$"

    data_type_name = "chinese_name"

    _inspect_level = 40

    pii = True

    def domain_verification(self, each_sample):
        def has_symbols(s):
            return bool(re.search(r"[^\w\s]", s))

        def has_english(s):
            return bool(re.search(r"[a-zA-Z]", s))

        def has_number(s):
            for char in s:
                if char.isdigit():
                    return True
            return False

        if has_number(each_sample):
            return False
        if has_english(each_sample):
            return False
        if has_symbols(each_sample):
            return False

        return True


# English Name
class EnglishNameInspector(RegexInspector):
    pattern = r"^([a-zA-Z]{2,}\s[a-zA-Z]{1,}'?-?[a-zA-Z]{2,}\s?([a-zA-Z]{1,})?)"

    data_type_name = "english_name"

    _inspect_level = 40

    pii = True

    name_min_length = 5
    """
    The min length of the name.

    GPT-4: The shortest full name in English could be something like "Ed Li" or "Al Lu", with just four characters including a space.
    """

    name_max_length = 70
    """
    The max length of the name.

    UK Government Data Standards Catalogue suggests 35 characters for each of Given Name and Family Name, or 70 characters for a single field to hold the Full Name.
    """

    def domain_verification(self, each_sample):
        def has_number(s):
            for char in s:
                if char.isdigit():
                    return True
            return False

        # English name should be between 5 - 70 characters
        if len(each_sample) > self.name_max_length:
            return False
        if len(each_sample) < self.name_min_length:
            return False
        # usually a name should not contains number
        if has_number(each_sample):
            return False
        return True


# 公司名
class ChineseCompanyNameInspector(RegexInspector):
    pattern = r".*?公司.*?"

    _match_percentage = 0.7

    data_type_name = "chinese_company_name"

    _inspect_level = 40

    pii = False


@hookimpl
def register(manager):
    manager.register("EmailInspector", EmailInspector)

    manager.register("ChinaMainlandIDInspector", ChinaMainlandIDInspector)

    manager.register("ChinaMainlandMobilePhoneInspector", ChinaMainlandMobilePhoneInspector)

    manager.register("ChinaMainlandPostCode", ChinaMainlandPostCode)

    manager.register("ChinaMainlandUnifiedSocialCreditCode", ChinaMainlandUnifiedSocialCreditCode)

    manager.register("ChinaMainlandAddressInspector", ChinaMainlandAddressInspector)

    manager.register("ChineseNameInspector", ChineseNameInspector)

    manager.register("EnglishNameInspector", EnglishNameInspector)

    manager.register("ChineseCompanyNameInspector", ChineseCompanyNameInspector)
