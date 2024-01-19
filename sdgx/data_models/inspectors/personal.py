from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.inspectors.regex import RegexInspector


class EmailInspector(RegexInspector):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    data_type_name = "email"


class ChinaMainlandIDInspector(RegexInspector):
    pattern = r"(^[1-9]\\d{5}(18|19|20)\\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\\d{3}[0-9Xx]$)|"

    data_type_name = "china_mainland_id"


class ChinaMainlandMobilePhoneInspector(RegexInspector):
    pattern = r"^1[3-9]\d{9}$"

    data_type_name = "china_mainland_mobile_phone"


@hookimpl
def register(manager):
    manager.register("EmailInspector", EmailInspector)

    manager.register("ChinaMainlandIDInspector", ChinaMainlandIDInspector)

    manager.register("ChinaMainlandMobilePhoneInspector", ChinaMainlandMobilePhoneInspector)
