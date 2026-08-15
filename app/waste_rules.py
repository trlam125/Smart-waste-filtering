from __future__ import annotations

from dataclasses import asdict, dataclass

from .class_schema import WASTE_CLASS_KEYS


@dataclass(frozen=True)
class WasteRule:
    key: str
    display_name: str
    category: str
    bin_name: str
    instruction: str
    icon: str

    def public_dict(self) -> dict[str, str]:
        return asdict(self)


# These keys MUST match the dataset directory names exactly.
# Dataset: SmartWaste_Household_EWaste_11class_native_v2
WASTE_RULES: tuple[WasteRule, ...] = (
    WasteRule(
        key="plastic_rigid",
        display_name="Nhựa cứng",
        category="Nhựa cứng / chai, hộp nhựa",
        bin_name="Điểm thu gom nhựa cứng nếu địa phương chấp nhận",
        instruction=(
            "Áp dụng cho chai, lọ, cốc, khay, hộp hoặc vật nhựa tương đối cứng. "
            "Làm rỗng, tráng sạch khi cần và để khô trước khi thu gom."
        ),
        icon="♻️",
    ),
    WasteRule(
        key="plastic_film",
        display_name="Nhựa mềm / màng nhựa",
        category="Túi, màng và bao bì nhựa mềm",
        bin_name="Điểm thu gom nhựa mềm nếu địa phương có hỗ trợ",
        instruction=(
            "Áp dụng cho túi nilon, màng bọc, túi PE/PP, gói mềm và bao bì nhựa mỏng. "
            "Làm rỗng, sạch và khô nếu có thể; ưu tiên điểm thu gom chuyên biệt."
        ),
        icon="🛍️",
    ),
    WasteRule(
        key="paper",
        display_name="Giấy",
        category="Giấy tái chế",
        bin_name="Thùng/điểm giấy tái chế nếu địa phương chấp nhận",
        instruction=(
            "Giữ giấy sạch và khô. Giấy dính nhiều dầu, thức ăn, phủ sáp hoặc vật liệu ghép "
            "nhiều lớp có thể cần xử lý theo quy định địa phương."
        ),
        icon="📄",
    ),
    WasteRule(
        key="cardboard",
        display_name="Bìa carton",
        category="Bìa carton tái chế",
        bin_name="Thùng/điểm carton nếu địa phương chấp nhận",
        instruction=(
            "Làm sạch, giữ khô và gấp phẳng hộp carton để giảm thể tích trước khi thu gom. "
            "Carton dính nhiều dầu hoặc thức ăn có thể không phù hợp để tái chế."
        ),
        icon="📦",
    ),
    WasteRule(
        key="metal",
        display_name="Kim loại",
        category="Kim loại tái chế",
        bin_name="Thùng/điểm kim loại tái chế nếu địa phương chấp nhận",
        instruction=(
            "Làm rỗng và vệ sinh lon/hộp kim loại thông thường. Bình xịt, hộp hóa chất hoặc "
            "vật chứa nguy hiểm cần theo hướng dẫn thu gom riêng."
        ),
        icon="🥫",
    ),
    WasteRule(
        key="glass",
        display_name="Thủy tinh",
        category="Thủy tinh tái chế",
        bin_name="Thùng/điểm thủy tinh nếu địa phương chấp nhận",
        instruction=(
            "Chai/lọ thủy tinh sạch có thể được thu gom tái chế tùy địa phương. Không tự động "
            "trộn gương, gốm sứ, bóng đèn hoặc thủy tinh chịu nhiệt vào cùng luồng."
        ),
        icon="🍾",
    ),
    WasteRule(
        key="organic",
        display_name="Rác hữu cơ",
        category="Rác hữu cơ",
        bin_name="Thùng rác hữu cơ",
        instruction="Loại bỏ bao bì và bỏ phần thức ăn, vỏ rau quả vào luồng rác hữu cơ/ủ compost nếu có.",
        icon="🍌",
    ),
    WasteRule(
        key="hazardous",
        display_name="Rác nguy hại",
        category="Rác nguy hại",
        bin_name="Điểm thu gom rác nguy hại",
        instruction=(
            "Không bỏ chung với rác sinh hoạt. Pin, bóng đèn, hóa chất, bình xịt hoặc vật nguy hại "
            "cần được mang đến điểm thu gom chuyên biệt."
        ),
        icon="⚠️",
    ),
    WasteRule(
        key="electronic",
        display_name="Rác điện tử",
        category="Rác điện tử",
        bin_name="Điểm thu gom rác điện tử",
        instruction=(
            "Không tự tháo linh kiện nguy hiểm. Mang điện thoại, bo mạch, phụ kiện máy tính và "
            "thiết bị điện tử đến điểm thu hồi hoặc tái chế điện tử."
        ),
        icon="🔌",
    ),
    WasteRule(
        key="textile",
        display_name="Dệt may / quần áo, giày dép",
        category="Rác dệt may",
        bin_name="Điểm thu gom/tái sử dụng dệt may nếu có",
        instruction=(
            "Ưu tiên tái sử dụng, quyên góp hoặc điểm thu gom dệt may cho quần áo, vải và giày dép "
            "còn phù hợp; phần hư hỏng xử lý theo quy định địa phương."
        ),
        icon="👕",
    ),
    WasteRule(
        key="other",
        display_name="Rác khác",
        category="Rác còn lại / vật liệu khác",
        bin_name="Thùng rác thông thường hoặc theo quy định địa phương",
        instruction=(
            "Dùng cho vật không thuộc 10 nhóm còn lại trong bộ dữ liệu. Nếu vật có thành phần nguy hại "
            "hoặc quy định thu gom riêng, ưu tiên hướng dẫn của địa phương."
        ),
        icon="🗑️",
    ),
)

RULE_BY_KEY = {rule.key: rule for rule in WASTE_RULES}
if tuple(RULE_BY_KEY) != WASTE_CLASS_KEYS:
    raise RuntimeError("WASTE_RULES must use the exact canonical 11-class dataset order.")

# All 11 classes are direct supervised classes in the trained dataset, including "other".
LEARNABLE_RULE_KEYS = frozenset(WASTE_CLASS_KEYS)
