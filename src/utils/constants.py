"""
Constants and Configuration
===========================

Predefined anatomical regions, thresholds, and labels.
"""

from ..types.models import TargetSize

# =============================================================================
# Anatomical Regions (normalized coordinates: [x1, y1, x2, y2])
# =============================================================================

ANATOMICAL_REGIONS = {
    # -------------------------------------------------------------------------
    # Chest X-ray regions
    # -------------------------------------------------------------------------
    "right_lung": [0.05, 0.15, 0.48, 0.85],
    "left_lung": [0.52, 0.15, 0.95, 0.85],
    "right_upper_lobe": [0.10, 0.15, 0.45, 0.45],
    "right_middle_lobe": [0.10, 0.35, 0.45, 0.55],
    "right_lower_lobe": [0.10, 0.45, 0.45, 0.85],
    "left_upper_lobe": [0.55, 0.15, 0.90, 0.45],
    "left_lower_lobe": [0.55, 0.45, 0.90, 0.85],
    "heart": [0.35, 0.40, 0.65, 0.80],
    "mediastinum": [0.35, 0.15, 0.65, 0.75],
    "carina": [0.40, 0.25, 0.60, 0.40],
    "costophrenic_angle_right": [0.05, 0.70, 0.30, 0.90],
    "costophrenic_angle_left": [0.70, 0.70, 0.95, 0.90],
    
    # -------------------------------------------------------------------------
    # Abdominal CT regions
    # -------------------------------------------------------------------------
    "liver": [0.50, 0.10, 0.95, 0.50],
    "spleen": [0.05, 0.10, 0.35, 0.40],
    "right_kidney": [0.55, 0.40, 0.85, 0.70],
    "left_kidney": [0.15, 0.40, 0.45, 0.70],
    "pancreas": [0.30, 0.30, 0.70, 0.55],
    "gallbladder": [0.50, 0.35, 0.70, 0.50],
    "stomach": [0.35, 0.20, 0.65, 0.45],
    "aorta": [0.40, 0.20, 0.55, 0.80],
    
    # -------------------------------------------------------------------------
    # Brain MRI regions
    # -------------------------------------------------------------------------
    "frontal_lobe": [0.20, 0.05, 0.80, 0.35],
    "parietal_lobe": [0.20, 0.30, 0.80, 0.55],
    "temporal_lobe_left": [0.05, 0.30, 0.35, 0.65],
    "temporal_lobe_right": [0.65, 0.30, 0.95, 0.65],
    "occipital_lobe": [0.25, 0.65, 0.75, 0.95],
    "cerebellum": [0.30, 0.75, 0.70, 0.95],
    "brainstem": [0.40, 0.60, 0.60, 0.85],
    "ventricles": [0.35, 0.35, 0.65, 0.55],
    
    # -------------------------------------------------------------------------
    # Spine regions
    # -------------------------------------------------------------------------
    "cervical_spine": [0.35, 0.05, 0.65, 0.25],
    "thoracic_spine": [0.35, 0.20, 0.65, 0.55],
    "lumbar_spine": [0.35, 0.50, 0.65, 0.80],
    "sacrum": [0.35, 0.75, 0.65, 0.95],
    
    # -------------------------------------------------------------------------
    # Generic quadrants (for any image)
    # -------------------------------------------------------------------------
    "upper_left": [0.0, 0.0, 0.5, 0.5],
    "upper_right": [0.5, 0.0, 1.0, 0.5],
    "lower_left": [0.0, 0.5, 0.5, 1.0],
    "lower_right": [0.5, 0.5, 1.0, 1.0],
    "center": [0.25, 0.25, 0.75, 0.75],
    "center_tight": [0.33, 0.33, 0.67, 0.67],
}


# =============================================================================
# Box Size Thresholds (max ratio of image area)
# =============================================================================

BOX_SIZE_THRESHOLDS = {
    TargetSize.TINY: 0.05,    # Max 5% of image (nodules, calcifications)
    TargetSize.SMALL: 0.15,   # Max 15% of image (lesions, small tumors)
    TargetSize.MEDIUM: 0.35,  # Max 35% of image (organs, larger tumors)
    TargetSize.LARGE: 0.60,   # Max 60% of image (full organs, diffuse)
}


# =============================================================================
# Modality Labels for BiomedCLIP
# =============================================================================

MODALITY_LABELS = [
    "X-ray",
    "CT scan", 
    "MRI",
    "Ultrasound",
    "Mammography",
    "PET scan",
    "Dermoscopy",
    "Fundoscopy",
    "Endoscopy",
    "Histopathology",
    "Microscopy",
]

ABNORMALITY_LABELS = [
    "Normal",
    "Abnormal",
    "Tumor/Mass",
    "Fracture",
    "Infection/Inflammation",
    "Hemorrhage",
    "Calcification",
    "Effusion",
    "Nodule",
    "Lesion",
]


# =============================================================================
# Fallback Region Mapping
# =============================================================================

FALLBACK_REGIONS = {
    "right_upper_lobe": ["right_lung", "upper_right"],
    "right_middle_lobe": ["right_lung", "center"],
    "right_lower_lobe": ["right_lung", "lower_right"],
    "left_upper_lobe": ["left_lung", "upper_left"],
    "left_lower_lobe": ["left_lung", "lower_left"],
    "right_lung": ["upper_right", "lower_right", "center"],
    "left_lung": ["upper_left", "lower_left", "center"],
    "liver": ["upper_right", "center"],
    "spleen": ["upper_left"],
    "right_kidney": ["lower_right", "center"],
    "left_kidney": ["lower_left", "center"],
    "frontal_lobe": ["upper_left", "upper_right", "center"],
    "occipital_lobe": ["lower_left", "lower_right", "center"],
    "cerebellum": ["lower_left", "lower_right"],
}


# =============================================================================
# Detection Prompts by Target Type
# =============================================================================

DETECTION_PROMPTS = {
    "nodule": "small nodule. round nodule. white spot. calcification. small mass.",
    "tumor": "tumor. mass. neoplasm. growth. lesion. abnormal mass.",
    "fracture": "fracture. break. crack. bone discontinuity. cortical break.",
    "hemorrhage": "hemorrhage. bleeding. blood. hematoma. hyperdense lesion.",
    "effusion": "effusion. fluid collection. pleural effusion. pericardial effusion.",
    "infection": "consolidation. infiltrate. opacity. pneumonia. abscess.",
    "calcification": "calcification. calcium deposit. calcified lesion.",
    "cyst": "cyst. fluid-filled. hypoechoic. anechoic.",
    "default": "abnormality. lesion. finding. pathology. suspicious area.",
}
