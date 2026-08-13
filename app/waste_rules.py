from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WasteRule:
    key: str
    display_name: str
    category: str
    bin_name: str
    instruction: str
    icon: str
    prompts: tuple[str, ...]

    def public_dict(self) -> dict[str, str]:
        data = asdict(self)
        data.pop("prompts", None)
        return data


# CLIP works better when prompts describe concrete visual appearances instead of
# only material names. Keep a similar number of prompts per direct class so one
# class does not receive a large prior advantage simply from having more labels.
#
# In Vietnamese daily usage, "nilon" often means thin flexible plastic bags/film,
# not nylon/polyamide textile. The dedicated ``nylon`` class below therefore maps
# to flexible plastic packaging, bags, film, pouches and wrappers.
WASTE_RULES: tuple[WasteRule, ...] = (
    WasteRule(
        key="plastic",
        display_name="Nhựa cứng / chai, hộp nhựa",
        category="Vật liệu có khả năng tái chế",
        bin_name="Thùng/điểm nhựa cứng nếu địa phương chấp nhận",
        instruction=(
            "Áp dụng cho chai, lọ, cốc, khay hoặc hộp nhựa tương đối cứng. Hãy làm rỗng, "
            "tráng sạch khi cần và làm khô trước khi thu gom. Không bỏ túi nilon, màng bọc "
            "hoặc bao bì nhựa mềm vào nhóm này; các vật đó được tách sang nhóm Nilon / nhựa mềm."
        ),
        icon="♻️",
        prompts=(
            "a clear PET plastic water bottle with a screw cap",
            "an opaque white plastic beverage bottle covered by a printed shrink sleeve",
            "a cylindrical rigid plastic drink bottle lying sideways on a table",
            "an upright rigid plastic beverage bottle with curved shoulders and a screw cap",
            "a rigid plastic food container tub tray or cup",
            "a molded hard plastic household container or detergent bottle",
        ),
    ),
    WasteRule(
        key="nylon",
        display_name="Nilon / nhựa mềm",
        category="Bao bì nhựa mềm",
        bin_name="Điểm thu gom nhựa mềm nếu địa phương có hỗ trợ",
        instruction=(
            "Áp dụng cho túi nilon/túi nhựa mỏng, màng bọc, túi PE/PP, gói snack và bao bì mềm. "
            "Làm rỗng, sạch và khô nếu có thể. Không mặc định bỏ chung với chai/hộp nhựa cứng: "
            "nhiều nơi không nhận nhựa mềm trong luồng tái chế thông thường, vì vậy hãy ưu tiên "
            "điểm thu gom chuyên biệt hoặc quy định của địa phương."
        ),
        icon="🛍️",
        prompts=(
            "a thin transparent plastic shopping bag with handles",
            "a crumpled polyethylene grocery bag made of flexible plastic film",
            "a soft plastic pouch or sachet with heat sealed edges",
            "a flexible plastic snack wrapper or food packaging packet",
            "a sheet of clear plastic film cling wrap or stretch wrap",
            "a lightweight flexible PE or PP plastic bag for recycling",
        ),
    ),
    WasteRule(
        key="paper",
        display_name="Giấy hoặc bìa carton",
        category="Vật liệu có khả năng tái chế",
        bin_name="Thùng/điểm giấy tái chế nếu địa phương chấp nhận",
        instruction=(
            "Chỉ đưa giấy/bìa sạch và khô vào luồng tái chế nếu nơi thu gom chấp nhận; gấp gọn carton. "
            "Giấy dính dầu, thức ăn, phủ sáp hoặc vật liệu ghép nhiều lớp có thể không tái chế được. "
            "Túi nhựa mỏng hoặc bao bì mềm có chữ in không được xem là giấy chỉ vì bề mặt có nhiều chữ."
        ),
        icon="📦",
        prompts=(
            "a brown corrugated cardboard shipping box with folded flaps",
            "a flattened corrugated cardboard sheet with visible fold lines",
            "discarded newspaper office paper or magazine pages",
            "a rectangular paperboard beverage carton or cereal box with straight folded edges",
            "a brown kraft paper bag made from paper with creased paper folds",
            "a rigid cardboard tube or paperboard container",
        ),
    ),
    WasteRule(
        key="metal",
        display_name="Kim loại",
        category="Vật liệu có khả năng tái chế",
        bin_name="Thùng/điểm kim loại tái chế nếu địa phương chấp nhận",
        instruction=(
            "Nếu là lon/hộp kim loại thông thường được nơi thu gom chấp nhận, hãy làm rỗng và làm sạch. "
            "Bình xịt, hộp hóa chất hoặc vật chứa nguy hiểm cần theo hướng dẫn thu gom riêng."
        ),
        icon="🥫",
        prompts=(
            "a discarded aluminum beverage can",
            "a steel food can or metal tin",
            "an empty metal drink can for recycling",
            "a clean aluminum food container or tray",
            "a small household object made mostly of metal",
            "an everyday object made mostly of aluminum or steel",
        ),
    ),
    WasteRule(
        key="glass",
        display_name="Thủy tinh",
        category="Vật liệu có khả năng tái chế",
        bin_name="Thùng/điểm thủy tinh nếu địa phương chấp nhận",
        instruction=(
            "Chai/lọ thủy tinh sạch có thể được thu gom tái chế tùy địa phương. Không tự động trộn gương, "
            "gốm sứ, bóng đèn hoặc thủy tinh chịu nhiệt vào cùng luồng; thủy tinh vỡ cần xử lý an toàn theo quy định."
        ),
        icon="🍾",
        prompts=(
            "a discarded clear glass bottle",
            "a colored glass beverage bottle",
            "an empty glass food jar",
            "a transparent glass container for recycling",
            "a household bottle or jar made of glass",
            "an everyday object made mostly of glass material",
        ),
    ),
    WasteRule(
        key="organic",
        display_name="Rác hữu cơ",
        category="Rác hữu cơ",
        bin_name="Thùng rác hữu cơ",
        instruction="Loại bỏ bao bì nhựa và bỏ phần thức ăn, vỏ rau quả vào thùng hữu cơ.",
        icon="🍌",
        prompts=(
            "discarded fruit peel or banana peel",
            "vegetable scraps from food preparation",
            "leftover cooked food on a plate",
            "food scraps for composting",
            "spoiled fruit or vegetable waste",
            "organic biodegradable kitchen waste",
        ),
    ),
    WasteRule(
        key="hazardous",
        display_name="Rác nguy hại",
        category="Rác nguy hại",
        bin_name="Điểm thu gom rác nguy hại",
        instruction="Không bỏ chung với rác sinh hoạt. Đem pin, bóng đèn hoặc hóa chất đến điểm thu gom chuyên biệt.",
        icon="⚠️",
        prompts=(
            "a discarded household battery requiring special collection",
            "a fluorescent light bulb or compact fluorescent lamp",
            "a container of toxic household chemicals",
            "a pesticide solvent or hazardous chemical container",
            "dangerous household waste requiring special collection",
            "toxic or hazardous waste that should not go in normal trash",
        ),
    ),
    WasteRule(
        key="electronic",
        display_name="Rác điện tử",
        category="Rác điện tử",
        bin_name="Điểm thu gom rác điện tử",
        instruction="Không tháo linh kiện nguy hiểm. Mang thiết bị đến điểm thu hồi hoặc tái chế điện tử.",
        icon="🔌",
        prompts=(
            "a discarded mobile phone or small electronic device",
            "a discarded charger power adapter or electronic cable",
            "a circuit board or electronic component",
            "a broken computer accessory or small appliance",
            "electronic equipment waiting for e-waste recycling",
            "electronic waste requiring specialized recycling",
        ),
    ),
    WasteRule(
        key="other",
        display_name="Rác còn lại",
        category="Rác còn lại",
        bin_name="Thùng rác thông thường",
        instruction=(
            "Dùng nhóm này khi vật không phù hợp các nhóm cụ thể hoặc địa phương không có luồng thu gom tương ứng. "
            "Nếu còn nghi ngờ, ưu tiên quy định phân loại tại địa phương thay vì cố đoán vật liệu."
        ),
        icon="🗑️",
        # Intentionally no CLIP prompts. "Other/general waste" is too broad and
        # tends to steal visually difficult recyclable objects. This category is
        # a fallback/guide category rather than a direct zero-shot competitor.
        prompts=(),
    ),
)

RULE_BY_KEY = {rule.key: rule for rule in WASTE_RULES}
DIRECT_RULES = tuple(rule for rule in WASTE_RULES if rule.key != "other")
LEARNABLE_RULE_KEYS = frozenset(rule.key for rule in DIRECT_RULES)
PROMPT_TO_KEY = {
    prompt: rule.key
    for rule in DIRECT_RULES
    for prompt in rule.prompts
}
ALL_PROMPTS = tuple(PROMPT_TO_KEY)
