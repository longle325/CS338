AFFORDANCE_PROMPTS = {
    "top clothes": "Place the garment naturally on the torso and shoulders with realistic sleeves, folds, neckline, and body fit.",
    "bottom clothes": "Place the garment on the waist, hips, and legs with natural drape, leg separation, and preserved footwear.",
    "dress": "Place the dress from shoulders or chest through the legs with coherent silhouette, waist fit, folds, and hem.",
    "shoe": "Place the shoes on both feet with correct left-right orientation, ground contact, scale, and perspective.",
    "earrings": "Place earrings at the ears, keep them small and sharp, and respect hair or face occlusion.",
    "bracelet": "Place the bracelet around the visible wrist with correct circular wrap, scale, and hand occlusion.",
    "necklace": "Place the necklace around the neck and upper chest with natural curve, highlights, and skin or clothing contact.",
    "ring": "Place the ring on a visible finger with correct tiny scale, metallic detail, and finger occlusion.",
    "sunglasses": "Place sunglasses across the eyes and nose bridge with symmetric lens alignment, temple arms, and face occlusion.",
    "glasses": "Place glasses across the eyes and nose bridge with transparent lenses, thin frame detail, and natural face occlusion.",
    "belt": "Place the belt around the waist with visible buckle alignment, correct horizontal wrap, and clothing occlusion.",
    "bag": "Place the bag near the hand, shoulder, or back according to the pose, with strap attachment and body occlusion.",
    "hat": "Place the hat on the head with correct scale, hair occlusion, brim perspective, and natural contact.",
    "tie": "Place the tie centered under the collar down the chest with correct knot, length, and shirt occlusion.",
    "bow tie": "Place the bow tie centered at the collar with symmetric wings, knot detail, and correct small scale.",
    "watch": "Place the watch around the wrist with correct scale, dial orientation, strap wrap, and hand occlusion.",
}


AFFORDANCE_BOXES = {
    "top clothes": (0.05, 0.18, 0.95, 0.72),
    "bottom clothes": (0.08, 0.42, 0.92, 0.98),
    "dress": (0.08, 0.18, 0.92, 0.98),
    "shoe": (0.05, 0.70, 0.95, 1.00),
    "earrings": (0.15, 0.04, 0.85, 0.42),
    "bracelet": (0.00, 0.30, 1.00, 0.85),
    "necklace": (0.22, 0.12, 0.78, 0.48),
    "ring": (0.00, 0.35, 1.00, 0.90),
    "sunglasses": (0.18, 0.05, 0.82, 0.36),
    "glasses": (0.18, 0.05, 0.82, 0.36),
    "belt": (0.12, 0.42, 0.88, 0.66),
    "bag": (0.00, 0.18, 1.00, 0.90),
    "hat": (0.12, 0.00, 0.88, 0.30),
    "tie": (0.28, 0.16, 0.72, 0.72),
    "bow tie": (0.25, 0.12, 0.75, 0.45),
    "watch": (0.00, 0.30, 1.00, 0.85),
}


HARD_CASE_CLASSES = {
    "ring",
    "earrings",
    "bracelet",
    "necklace",
    "watch",
    "glasses",
    "sunglasses",
    "bag",
    "shoe",
    "hat",
    "tie",
    "bow tie",
}


CLASS_ALIASES = {
    "bottom cloth": "bottom clothes",
    "bottom clothing": "bottom clothes",
    "top cloth": "top clothes",
    "top clothing": "top clothes",
    "shoes": "shoe",
    "earring": "earrings",
    "bowtie": "bow tie",
    "sunglass": "sunglasses",
    "backpack": "bag",
    "shoulder bag": "bag",
    "tote": "bag",
}


def normalize_class_name(name: str) -> str:
    normalized = (name or "").strip().lower().replace("_", " ")
    return CLASS_ALIASES.get(normalized, normalized)


def class_index_map():
    classes = sorted(AFFORDANCE_BOXES)
    return {name: index for index, name in enumerate(classes)}
