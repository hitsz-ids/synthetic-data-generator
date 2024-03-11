import re

from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.inspectors.regex import RegexInspector


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
        if len(each_sample) < self.address_min_length: return False
        if len(each_sample) > self.address_max_length: return False
        # notice to distinguishing from the company name
        if each_sample.endswith("公司"):
            return False
        return True


@hookimpl
def register(manager):
    manager.register("EmailInspector", EmailInspector)

    manager.register("ChinaMainlandIDInspector", ChinaMainlandIDInspector)

    manager.register("ChinaMainlandMobilePhoneInspector", ChinaMainlandMobilePhoneInspector)

    manager.register("ChinaMainlandPostCode", ChinaMainlandPostCode)

    manager.register("ChinaMainlandUnifiedSocialCreditCode", ChinaMainlandUnifiedSocialCreditCode)

    manager.register("ChinaMainlandAddressInspector", ChinaMainlandAddressInspector)
