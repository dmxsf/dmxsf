"""The tickets, prompts and word problems of the prompt-engineering page (one set per language).

Copied from the page's code blocks so that the lab and the page run on the same text. The tickets
and their store are invented: 24 tickets, 8 per class, 12 for development ("dev") and 12 held out
("test"), with a few boundary pairs on purpose. The Chinese and English sets are written separately,
not translated word for word, and the page ran each of them through the same model.
"""

from __future__ import annotations

ZH = {
    "dev": [
        ("你们支持七天无理由退货吗？", "faq"),
        ("怎么修改默认收货地址？", "faq"),
        ("退款一般多久到账？", "faq"),
        ("会员积分怎么抵现金？", "faq"),
        ("订单 A1001 现在到哪了？", "order_lookup"),
        ("我昨天下的单，什么时候能发货？", "order_lookup"),
        ("帮我看看订单 B2002 的退款处理到哪一步了。", "order_lookup"),
        ("我买的蓝牙耳机物流显示在运输中，预计哪天送到？订单号 C3003。", "order_lookup"),
        ("订单已扣款但还是显示未支付，怎么办？", "escalation"),
        ("我的账户被盗刷了三笔，立刻处理！", "escalation"),
        ("你们客服一直踢皮球，再不解决我就去消协投诉。", "escalation"),
        ("收到的充电宝充电时发烫鼓包，我很担心安全。", "escalation"),
    ],
    "test": [
        ("发票可以开增值税专用发票吗？", "faq"),
        ("预售商品是不是不能退？", "faq"),
        ("海外地址能下单吗？运费怎么算？", "faq"),
        ("怎么取消还没发货的订单？", "faq"),
        ("麻烦查一下订单 D4004 有没有出库。", "order_lookup"),
        ("我上周买的键盘，快递单号是多少？", "order_lookup"),
        ("订单 E5005 的退货申请审核通过了吗？", "order_lookup"),
        ("订单 F6006 的快递现在到哪个城市了？", "order_lookup"),
        ("同一笔订单我被扣了两次钱，请马上退回多扣的那笔。", "escalation"),
        ("我没下过单，却收到了扣款短信，是不是被盗了？", "escalation"),
        ("商品和描述严重不符，我已经咨询律师，准备起诉你们。", "escalation"),
        ("宝宝用了你们卖的奶瓶后手被烫伤，要求赔偿并追究责任。", "escalation"),
    ],
    "shots": [
        ("运费险是怎么赔付的？", "faq"),
        ("订单 G7007 的包裹签收了吗？", "order_lookup"),
        ("退款到账的金额比应退的少了 50 元，客服一直不处理。", "escalation"),
    ],
    "math": [
        ("小明有 23 个苹果，送给同学 8 个，又买了 3 盒，每盒 12 个。他现在有多少个苹果？", 51),
        ("一个书架有 4 层，每层放 18 本书。借出 15 本后，又收回 6 本。书架上现在有多少本书？", 63),
        (
            "食堂早上做了 120 个包子，上午卖出了其中的 3/4，下午又做了 30 个。现在一共有多少个包子？",
            60,
        ),
        (
            "一辆车先以每小时 60 千米开了 2.5 小时，休息后又以每小时 80 千米开了 1.5 小时。一共行驶多少千米？",
            270,
        ),
        ("一件标价 200 元的外套打八折，之后又用了一张 30 元的优惠券。最终要付多少元？", 130),
        ("班里有 45 名学生，男生比女生多 7 人。女生有多少人？", 19),
        ("小红每天读 14 页书，读了 6 天后还剩 59 页没读。这本书一共有多少页？", 143),
        ("一箱牛奶有 24 盒，买了 5 箱，喝掉 37 盒，再送给邻居 2 箱。还剩多少盒？", 35),
        (
            "水池里有 500 升水，每分钟流出 15 升，同时每分钟注入 8 升。20 分钟后水池里有多少升水？",
            360,
        ),
        ("三个数的平均数是 24，其中两个数分别是 20 和 31。第三个数是多少？", 21),
        ("小华已有 150 元，每周存入 25 元并花掉 10 元。12 周后他有多少元？", 330),
        ("长方形菜地长 18 米、宽 12 米，沿周长每隔 3 米种一棵树。一共种多少棵树？", 20),
    ],
    "v1": "你是电商客服分流助手。读取用户工单，输出 JSON，字段：task_type（faq、order_lookup、escalation 三选一）和 requires_human（true 或 false）。",
    "rules": "\n分类规则：\n- faq：询问通用规则、政策或操作方法，与用户自己的某笔订单无关。\n- order_lookup：询问自己某笔订单、退款或物流的进度与信息，且没有异常。\n- escalation：涉及资金异常（重复扣款、被盗刷、扣款未生效）、人身安全、投诉或法律纠纷。\nrequires_human 仅在 escalation 时为 true。\n",
    "strict": "只输出一个 JSON 对象，不要代码围栏，不要解释。",
    "cot": "先用一两句话分析工单属于哪一类，再在最后一行输出一个 JSON 对象，不要代码围栏。",
    "sys_direct": "你是数学助手。只输出最终答案的数字，不要写单位和任何解释。",
    "sys_cot": "你是数学助手。请逐步推理，最后一行写“答案：”加数字。",
    "render_with": "<|im_start|>system\n你是客服分流助手。只输出 JSON。<|im_end|>\n<|im_start|>user\n订单已扣款但还是显示未支付，怎么办？<|im_end|>\n<|im_start|>assistant\n",
    "render_without": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n订单已扣款但还是显示未支付，怎么办？<|im_end|>\n<|im_start|>assistant\n",
    "demo_system": "你是客服分流助手。只输出 JSON。",
    "demo_ticket": "订单已扣款但还是显示未支付，怎么办？",
}

EN = {
    "dev": [
        ("Do you accept returns within seven days without a reason?", "faq"),
        ("How do I change my default shipping address?", "faq"),
        ("How long does a refund usually take to arrive?", "faq"),
        ("How can I redeem loyalty points for cash?", "faq"),
        ("Where is order A1001 right now?", "order_lookup"),
        ("I ordered yesterday. When will it ship?", "order_lookup"),
        ("Can you check how far along the refund for order B2002 is?", "order_lookup"),
        (
            "The tracking for my Bluetooth earbuds says in transit. When will they arrive? Order C3003.",
            "order_lookup",
        ),
        ("My card was charged but the order still shows unpaid. What do I do?", "escalation"),
        ("Three unauthorized charges hit my account. Fix this right now!", "escalation"),
        (
            "Support keeps passing me around. Solve this or I will file a complaint with the consumer protection agency.",
            "escalation",
        ),
        (
            "The power bank I received gets very hot and the case is bulging. I am worried about safety.",
            "escalation",
        ),
    ],
    "test": [
        ("Can I get a VAT invoice for my purchase?", "faq"),
        ("Are pre-order items non-refundable?", "faq"),
        ("Do you ship overseas, and how is shipping calculated?", "faq"),
        ("How do I cancel an order that has not shipped yet?", "faq"),
        ("Please check whether order D4004 has left the warehouse.", "order_lookup"),
        ("What is the tracking number for the keyboard I bought last week?", "order_lookup"),
        ("Has the return request for order E5005 been approved?", "order_lookup"),
        ("Which city is the parcel for order F6006 in now?", "order_lookup"),
        (
            "I was charged twice for the same order. Please refund the extra charge immediately.",
            "escalation",
        ),
        (
            "I never placed an order but got a payment text message. Was my account stolen?",
            "escalation",
        ),
        (
            "The product is nothing like the description. I have talked to a lawyer and plan to sue.",
            "escalation",
        ),
        (
            "My baby burned a hand on the bottle you sell. I demand compensation and accountability.",
            "escalation",
        ),
    ],
    "shots": [
        ("How does shipping insurance pay out?", "faq"),
        ("Has the parcel for order G7007 been delivered?", "order_lookup"),
        ("The refund I received is 50 dollars short, and support keeps ignoring it.", "escalation"),
    ],
    "math": [
        (
            "Ben has 23 apples, gives 8 to a classmate, then buys 3 boxes with 12 apples each. How many apples does he have now?",
            51,
        ),
        (
            "A bookshelf has 4 shelves with 18 books on each. After 15 books are borrowed, 6 are returned. How many books are on the shelf now?",
            63,
        ),
        (
            "The canteen made 120 buns in the morning and sold 3/4 of them. In the afternoon it made 30 more. How many buns are there now?",
            60,
        ),
        (
            "A car drives at 60 km/h for 2.5 hours, rests, then drives at 80 km/h for 1.5 hours. How many kilometers does it cover in total?",
            270,
        ),
        (
            "A jacket is priced at 200 dollars with a 20 percent discount, and then a 30 dollar coupon is applied. How many dollars does the buyer finally pay?",
            130,
        ),
        (
            "A class has 45 students, and there are 7 more boys than girls. How many girls are there?",
            19,
        ),
        (
            "Amy reads 14 pages a day. After 6 days, 59 pages are still unread. How many pages does the book have?",
            143,
        ),
        (
            "A case of milk holds 24 cartons. Sam buys 5 cases, drinks 37 cartons, and gives 2 cases to a neighbor. How many cartons are left?",
            35,
        ),
        (
            "A tank holds 500 liters. Every minute 15 liters flow out and 8 liters flow in. How many liters are in the tank after 20 minutes?",
            360,
        ),
        (
            "The average of three numbers is 24, and two of them are 20 and 31. What is the third number?",
            21,
        ),
        (
            "Leo has 150 dollars. Each week he saves 25 dollars and spends 10 dollars. How many dollars does he have after 12 weeks?",
            330,
        ),
        (
            "A rectangular garden is 18 m long and 12 m wide. A tree is planted every 3 m along the perimeter. How many trees are planted?",
            20,
        ),
    ],
    "v1": "You triage e-commerce support tickets. Read the ticket and output JSON with the fields task_type (one of faq, order_lookup, escalation) and requires_human (true or false).",
    "rules": "\nClassification rules:\n- faq: asks about general rules, policies or how-to, not tied to the user's own order.\n- order_lookup: asks for the status or details of the user's own order, refund or shipment, with nothing abnormal.\n- escalation: involves a money problem (double charge, unauthorized charge, charged but unpaid), physical safety, a complaint or a legal dispute.\nrequires_human is true only for escalation.\n",
    "strict": "Output a single JSON object, with no code fence and no explanation.",
    "cot": "Think step by step. Start with 'Analysis:' and explain in one or two sentences which kind of ticket it is. Then, on the last line, output a single JSON object, with no code fence.",
    "sys_direct": "You are a math assistant. Output only the final numeric answer, with no units and no explanation.",
    "sys_cot": "You are a math assistant. Reason step by step, and write the last line as 'Answer: ' followed by the number.",
    "render_with": "<|im_start|>system\nYou triage tickets. Output JSON only.<|im_end|>\n<|im_start|>user\nMy card was charged but the order still shows unpaid.<|im_end|>\n<|im_start|>assistant\n",
    "render_without": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nMy card was charged but the order still shows unpaid.<|im_end|>\n<|im_start|>assistant\n",
    "demo_system": "You triage tickets. Output JSON only.",
    "demo_ticket": "My card was charged but the order still shows unpaid.",
}

DATA = {"zh": ZH, "en": EN}
