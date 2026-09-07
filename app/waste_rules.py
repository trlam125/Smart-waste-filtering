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
        display_name="Rigid plastic",
        category="Rigid plastic / bottles and containers",
        bin_name="Rigid plastic collection point if accepted locally",
        instruction=(
            "Applies to bottles, jars, cups, trays, containers, and other relatively rigid plastic items. "
            "Empty the item, rinse it when needed, and let it dry before collection."
        ),
        icon="♻️",
    ),
    WasteRule(
        key="plastic_film",
        display_name="Soft plastic / plastic film",
        category="Bags, film, and flexible plastic packaging",
        bin_name="Soft plastic collection point if available locally",
        instruction=(
            "Applies to plastic bags, wrap, PE/PP bags, flexible pouches, and thin plastic packaging. "
            "Empty, clean, and dry the material when possible; use a specialized collection point when available."
        ),
        icon="🛍️",
    ),
    WasteRule(
        key="paper",
        display_name="Paper",
        category="Recyclable paper",
        bin_name="Paper recycling bin or collection point if accepted locally",
        instruction=(
            "Keep paper clean and dry. Paper heavily contaminated with oil or food, wax-coated paper, or laminated "
            "multi-layer materials may require disposal according to local rules."
        ),
        icon="📄",
    ),
    WasteRule(
        key="cardboard",
        display_name="Cardboard",
        category="Recyclable cardboard",
        bin_name="Cardboard recycling bin or collection point if accepted locally",
        instruction=(
            "Clean and keep cardboard dry, then flatten boxes to reduce volume before collection. "
            "Cardboard heavily contaminated with oil or food may not be suitable for recycling."
        ),
        icon="📦",
    ),
    WasteRule(
        key="metal",
        display_name="Metal",
        category="Recyclable metal",
        bin_name="Metal recycling bin or collection point if accepted locally",
        instruction=(
            "Empty and clean ordinary metal cans and containers. Aerosol cans, chemical containers, or "
            "hazardous containers require separate collection guidance."
        ),
        icon="🥫",
    ),
    WasteRule(
        key="glass",
        display_name="Glass",
        category="Recyclable glass",
        bin_name="Glass recycling bin or collection point if accepted locally",
        instruction=(
            "Clean glass bottles and jars may be recyclable depending on local rules. Do not automatically "
            "mix mirrors, ceramics, light bulbs, or heat-resistant glass into the same recycling stream."
        ),
        icon="🍾",
    ),
    WasteRule(
        key="organic",
        display_name="Organic waste",
        category="Organic waste",
        bin_name="Organic waste bin",
        instruction="Remove packaging and place food scraps and fruit or vegetable peels in the organic waste or compost stream if available.",
        icon="🍌",
    ),
    WasteRule(
        key="hazardous",
        display_name="Hazardous waste",
        category="Hazardous waste",
        bin_name="Hazardous waste collection point",
        instruction=(
            "Do not mix with household waste. Batteries, light bulbs, chemicals, aerosol cans, and other hazardous items "
            "should be taken to a specialized collection point."
        ),
        icon="⚠️",
    ),
    WasteRule(
        key="electronic",
        display_name="Electronic waste",
        category="Electronic waste",
        bin_name="Electronic waste collection point",
        instruction=(
            "Do not dismantle hazardous components yourself. Take phones, circuit boards, computer accessories, and "
            "electronic devices to an electronics take-back or recycling point."
        ),
        icon="🔌",
    ),
    WasteRule(
        key="textile",
        display_name="Textiles / clothing and footwear",
        category="Textile waste",
        bin_name="Textile reuse or collection point if available",
        instruction=(
            "Prioritize reuse, donation, or textile collection points for clothing, fabric, and footwear "
            "that are still usable; dispose of damaged items according to local rules."
        ),
        icon="👕",
    ),
    WasteRule(
        key="other",
        display_name="Other waste",
        category="Residual waste / other materials",
        bin_name="General waste bin or local designated disposal",
        instruction=(
            "Use for items that do not belong to the other 10 dataset categories. If an item contains hazardous components "
            "or has special collection requirements, follow local guidance first."
        ),
        icon="🗑️",
    ),
)

RULE_BY_KEY = {rule.key: rule for rule in WASTE_RULES}
if tuple(RULE_BY_KEY) != WASTE_CLASS_KEYS:
    raise RuntimeError("WASTE_RULES must use the exact canonical 11-class dataset order.")

# All 11 classes are direct supervised classes in the trained dataset, including "other".
LEARNABLE_RULE_KEYS = frozenset(WASTE_CLASS_KEYS)
