import re 

from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.inspectors.regex import RegexInspector


class EmailInspector(RegexInspector):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    data_type_name = "email"

    _inspect_level = 30

    pii = True


class ChinaMainlandIDInspector(RegexInspector):
    pattern = r"^[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$"
    

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

    pattern_ID = r"^[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$"

    p_id = re.compile(pattern_ID)
    
    def domain_verification(self, each_sample):
        if re.match(self.p_id, each_sample):
            return False
        return True
    

@hookimpl
def register(manager):
    manager.register("EmailInspector", EmailInspector)

    manager.register("ChinaMainlandIDInspector", ChinaMainlandIDInspector)

    manager.register("ChinaMainlandMobilePhoneInspector", ChinaMainlandMobilePhoneInspector)

    manager.register("ChinaMainlandPostCode", ChinaMainlandPostCode)

    manager.register("ChinaMainlandUnifiedSocialCreditCode", ChinaMainlandUnifiedSocialCreditCode)
