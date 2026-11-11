"""The small corpora, queries and sample documents of the RAG page (one set per language).

Everything here is invented for the demonstration: a made-up company handbook of 23 chunks in six
topics, 12 queries labelled with the one chunk each should hit, and the made-up expense policy that
the page's chunking widget cuts up. The Chinese and English sets are separate texts, not translations
of each other word for word, because each page was run with its own corpus and models.
"""

from __future__ import annotations

CHUNKS_ZH = {
    "exp-scope": "差旅报销范围：员工因公出差产生的交通、住宿和餐饮费用可以报销，机票须选经济舱，高铁须选二等座。",
    "exp-limit": "住宿费按城市分级设置上限，一线城市每晚不超过 600 元，其他城市每晚不超过 400 元。",
    "exp-submit": "报销需在出差结束后 30 个自然日内提交，逾期未提交的费用将不予报销。",
    "exp-material": "提交报销时需上传电子发票、行程单和审批单，缺少任一材料会被退回补充。",
    "exp-advance": "借款需提前 3 个工作日申请，单次不超过 10000 元，出差结束后 7 个工作日内必须冲销。",
    "acc-reset": "密码重置：在自助门户点击“找回密码”，通过手机验证码验证身份后即可设置新密码。",
    "acc-policy": "《信息安全管理制度》IT-SEC-003 规定：密码至少 12 位，须同时包含大小写字母、数字和符号。",
    "acc-lock": "账号连续输错密码 5 次会被锁定 30 分钟，期间无法登录，30 分钟后自动解锁。",
    "acc-mfa": "所有员工必须在入职一周内启用二次验证，可使用验证器应用或硬件安全密钥。",
    "vpn-install": "远程办公访问内网须先安装公司 VPN 客户端，并使用公司账号和二次验证登录。",
    "vpn-cert": "VPN 提示“证书已过期”时，需要在自助门户重新下载证书并导入客户端，有效期为一年。",
    "remote-policy": "经部门负责人批准，员工每周最多可远程办公 2 天，远程办公当天须保持在线并按时参加例会。",
    "leave-annual": "年假按累计工作年限计算：满 1 年不满 10 年为 5 天，满 10 年不满 20 年为 10 天，满 20 年为 15 天。",
    "leave-sick": "病假需提交医院出具的诊断证明，三天以内由直属主管批准，超过三天须报人事部备案，详见 HR-LV-012。",
    "attendance": "公司实行弹性打卡，核心工作时段为 10:00 至 16:00，每日须工作满 8 小时。",
    "comp-time": "休息日加班可申请调休，调休须在加班后 3 个月内休完，逾期自动作废。",
    "err-1041": "E1041：登录会话超时，用户需要重新登录；会话默认有效期为 8 小时。",
    "err-1042": "E1042：数据库连接池耗尽，请检查最大连接数配置并重启受影响的服务。",
    "err-1043": "E1043：磁盘使用率超过 90%，请清理日志或扩容数据盘。",
    "err-1044": "E1044：服务端证书即将过期，请在到期前 14 天内完成证书续签。",
    "err-1045": "E1045：请求被限流，同一接口每分钟最多调用 100 次，请稍后重试。",
    "buy-threshold": "单笔采购金额超过 5000 元须提供至少三家供应商的比价单，并经部门负责人审批。",
    "backup": "生产数据库每日凌晨 2 点做全量备份，备份保留 30 天，恢复演练每季度进行一次。",
}
QUERIES_ZH = [
    ("出差回来之后，单据最晚什么时候要交？", "exp-submit"),
    ("登录口令忘了怎么办？", "acc-reset"),
    ("在家办公时怎么连上公司内网？", "vpn-install"),
    ("身体不舒服请假要带什么证明？", "leave-sick"),
    ("E1042", "err-1042"),
    ("错误码 E1044 怎么处理", "err-1044"),
    ("IT-SEC-003 对密码长度有什么要求", "acc-policy"),
    ("HR-LV-012", "leave-sick"),
    ("数据库连不上了，报错说连接池满了", "err-1042"),
    ("报销需要上传哪些材料？", "exp-material"),
    ("借款最多能借多少，什么时候要还？", "exp-advance"),
    ("输错几次密码会被锁？", "acc-lock"),
]

CHUNKS_EN = {
    "exp-scope": (
        "Travel expense scope: transport, lodging and meal costs on "
        "business trips are reimbursable; flights must be economy "
        "class and high-speed rail second class."
    ),
    "exp-limit": (
        "Lodging is capped by city tier: at most 600 yuan per night "
        "in tier-one cities and at most 400 yuan per night elsewhere."
    ),
    "exp-submit": (
        "Expense claims must be submitted within 30 calendar days "
        "after the trip ends; late claims will not be reimbursed."
    ),
    "exp-material": (
        "A claim needs an electronic invoice, the itinerary and the "
        "approval form; a claim missing any of them is sent back."
    ),
    "exp-advance": (
        "An advance must be requested 3 working days ahead, up to "
        "10000 yuan per request, and settled within 7 working days "
        "after the trip."
    ),
    "acc-reset": (
        'Password reset: click "Forgot password" in the self-service '
        "portal and verify your identity with an SMS code to set a "
        "new password."
    ),
    "acc-policy": (
        "Information Security Policy IT-SEC-003 requires passwords of"
        " at least 12 characters with upper and lower case letters, "
        "digits and symbols."
    ),
    "acc-lock": (
        "An account is locked for 30 minutes after 5 consecutive "
        "wrong passwords and unlocks automatically afterwards."
    ),
    "acc-mfa": (
        "Every employee must enable two-factor authentication within "
        "the first week, using an authenticator app or a hardware "
        "security key."
    ),
    "vpn-install": (
        "Remote access to the intranet requires the company VPN "
        "client, signing in with your company account and two-factor "
        "authentication."
    ),
    "vpn-cert": (
        'When the VPN reports "certificate expired", download a new '
        "certificate from the self-service portal and import it into "
        "the client; certificates last one year."
    ),
    "remote-policy": (
        "With the approval of the department head, employees may work"
        " remotely up to 2 days per week and must stay online and "
        "join the regular meetings."
    ),
    "leave-annual": (
        "Annual leave depends on total years of work: 5 days for 1 to"
        " under 10 years, 10 days for 10 to under 20 years and 15 "
        "days for 20 years or more."
    ),
    "leave-sick": (
        "Sick leave needs a diagnosis certificate from a hospital; up"
        " to three days is approved by the direct manager, longer "
        "leave must be filed with HR (see HR-LV-012)."
    ),
    "attendance": (
        "The company uses flexible hours with core hours from 10:00 "
        "to 16:00; everyone must work 8 hours a day."
    ),
    "comp-time": (
        "Overtime on rest days can be taken as time off in lieu, "
        "which must be used within 3 months of the overtime, or it "
        "expires."
    ),
    "err-1041": (
        "E1041: login session timed out and the user must sign in "
        "again; sessions are valid for 8 hours by default."
    ),
    "err-1042": (
        "E1042: database connection pool exhausted; check the maximum"
        " connections setting and restart the affected service."
    ),
    "err-1043": ("E1043: disk usage above 90%; clean up logs or expand the data disk."),
    "err-1044": (
        "E1044: the server certificate is about to expire; renew it within 14 days before expiry."
    ),
    "err-1045": (
        "E1045: request rate limited; each endpoint accepts at most "
        "100 calls per minute, so retry later."
    ),
    "buy-threshold": (
        "A single purchase above 5000 yuan needs quotes from at least"
        " three suppliers and approval from the department head."
    ),
    "backup": (
        "The production database gets a full backup at 2 a.m. every "
        "day, kept for 30 days, with a recovery drill every quarter."
    ),
}
QUERIES_EN = [
    (("How late can I hand in receipts after coming back from a trip?"), "exp-submit"),
    ("I forgot my login credentials, what now?", "acc-reset"),
    ("How do I reach the company network from home?", "vpn-install"),
    ("What do I need to bring for a day off when I feel ill?", "leave-sick"),
    ("E1042", "err-1042"),
    ("What should I do about error code E1044?", "err-1044"),
    ("How long must passwords be under IT-SEC-003?", "acc-policy"),
    ("HR-LV-012", "leave-sick"),
    (("Can't reach the database, the error says the connection pool is full"), "err-1042"),
    ("Which documents must I upload with an expense claim?", "exp-material"),
    (("How much can I borrow in advance and when must I pay it back?"), "exp-advance"),
    ("How many wrong passwords before I get locked out?", "acc-lock"),
]

SAMPLE_DOC_ZH = """\
差旅报销制度

一、报销范围
员工因公出差产生的交通、住宿和餐饮费用可以报销。交通费按实际发生金额报销，乘坐飞机须选择经济舱，高铁须选择二等座。住宿费按城市分级设置上限，一线城市每晚不超过 600 元，其他城市每晚不超过 400 元。

二、提交流程
报销需在出差结束后 30 个自然日内提交，逾期未提交的费用将不予报销。提交时需上传电子发票、行程单和审批单，缺少任一材料会被退回补充。财务部在收到完整材料后 5 个工作日内完成审核并付款。

三、借款与结算
借款需提前 3 个工作日申请，单次借款不超过 10000 元。出差结束后 7 个工作日内必须冲销借款，未冲销前不得再次借款。

四、特殊情况
因客户临时要求而超出住宿上限的，须在出差前取得部门负责人的书面同意。出国出差的标准另行制定，不适用本制度。"""

SAMPLE_DOC_EN = """\
Travel Expense Policy

1. Scope
Transport, lodging and meal costs incurred on business trips can be reimbursed. Transport is reimbursed at the actual amount, with economy class for flights and second class for high-speed rail. Lodging is capped by city tier: at most 600 yuan per night in tier-one cities and at most 400 yuan per night elsewhere.

2. Submission
Claims must be submitted within 30 calendar days after the trip ends, and late claims will not be reimbursed. An electronic invoice, the itinerary and the approval form must be uploaded, and a claim missing any of them is sent back. Finance reviews a complete claim and pays within 5 working days of receiving it.

3. Advances and settlement
An advance must be requested 3 working days ahead and may not exceed 10000 yuan per request. It must be settled within 7 working days after the trip ends, and no new advance is issued until then.

4. Exceptions
Lodging above the cap, when a client changes plans at short notice, needs the written consent of the department head before the trip. Standards for overseas trips are set separately and do not fall under this policy."""

CHUNKS = {"zh": CHUNKS_ZH, "en": CHUNKS_EN}
QUERIES = {"zh": QUERIES_ZH, "en": QUERIES_EN}
SAMPLE_DOCS = {"zh": SAMPLE_DOC_ZH, "en": SAMPLE_DOC_EN}
