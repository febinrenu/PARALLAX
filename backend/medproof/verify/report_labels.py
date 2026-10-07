"""Turn a second reader's free-text report into labels in our vocabulary, with negation and hedging.

Deterministic on purpose: the same report always gives the same labels, nothing is sent to an LLM,
and every label carries the sentence it came from for the audit view.

Label names follow each modality's classifier: TorchXRayVision names for chest X-ray, HAM10000
codes for dermoscopy, `fracture` for bone and `glioma/meningioma/pituitary/notumor` for brain MRI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# label -> (strong terms, weak terms). A weak term only ever gives a hedged label.
Vocab = dict[str, tuple[tuple[str, ...], tuple[str, ...]]]

CXR: Vocab = {
    "Atelectasis": (("atelectasis", "atelectatic", "lung collapse", "collapse"), ()),
    "Consolidation": (("consolidation", "consolidative"), ()),
    "Infiltration": (("infiltrate", "infiltrates", "infiltration"), ()),
    "Pneumothorax": (("pneumothorax", "pneumothoraces"), ()),
    "Edema": (("pulmonary edema", "edema", "oedema"), ("vascular congestion", "interstitial markings")),
    "Emphysema": (("emphysema",), ("hyperinflat\\w*", "flattened diaphragm\\w*")),
    "Fibrosis": (("fibrosis", "fibrotic"), ("scarring",)),
    "Effusion": (("pleural effusions?", "effusions?"), ("blunting of the costophrenic",)),
    "Pneumonia": (("pneumonia", "pneumonic"), ()),
    "Pleural_Thickening": (("pleural thickening",), ()),
    "Cardiomegaly": (
        ("cardiomegaly", "enlarged heart", "heart (?:size )?is (?:mildly |moderately |markedly )?enlarged",
         "enlarged cardiac silhouette", "cardiac enlargement"), (),
    ),
    "Nodule": (("nodules?", "nodular"), ()),
    "Mass": (("mass", "masses"), ()),
    "Hernia": (("hernia", "hiatal"), ()),
    "Lung Lesion": (("lung lesions?", "pulmonary lesions?"), ()),
    "Fracture": (("fractures?", "fractured"), ()),
    "Lung Opacity": (("airspace opacit\\w+", "lung opacit\\w+", "opacit\\w+", "airspace disease"), ()),
    "Enlarged Cardiomediastinum": (
        ("widened mediastinum", "mediastinal widening", "enlarged cardiomediastin\\w+", "widening of the mediastinum"), (),
    ),
}
SKIN: Vocab = {
    "akiec": (("actinic keratos\\w+", "solar keratos\\w+", "bowen\\w*", "intraepithelial carcinoma"), ()),
    "bcc": (("basal cell carcinoma", "basal cell", "bcc"), ()),
    "bkl": (("seborrh?eic keratos\\w+", "benign keratos\\w+", "solar lentigo", "lentigo", "lichen planus-like keratosis", "keratosis"), ()),
    "df": (("dermatofibroma\\w*", "histiocytoma"), ()),
    "mel": (("melanoma\\w*",), ()),
    "nv": (("melanocytic nevus", "melanocytic naevus", "nevus", "nevi", "naevus", "mole"), ()),
    "vasc": (("hemangioma\\w*", "haemangioma\\w*", "angioma\\w*", "vascular lesion", "angiokeratoma", "pyogenic granuloma", "cherry"), ()),
}
BONE: Vocab = {
    "fracture": (("fractures?", "fractured", "broken", "cortical break", "cortical disruption"), ("cortical irregularity",)),
}
BRAIN: Vocab = {
    "glioma": (("glioma\\w*", "glioblastoma", "astrocytoma", "gbm"), ()),
    "meningioma": (("meningioma\\w*",), ()),
    "pituitary": (("pituitary", "macroadenoma", "microadenoma", "sellar adenoma"), ()),
}
VOCABS: dict[str, Vocab] = {"cxr": CXR, "skin_dermoscopy": SKIN, "bone_xray": BONE, "brain_mri": BRAIN}

# P4's core/vocab.py names the skin classes in full; the dataset codes above are what the classifier and this parser use.
_SKIN_ALIASES = {
    "melanoma": "mel", "melanocytic_nevus": "nv", "basal_cell_carcinoma": "bcc", "actinic_keratosis": "akiec",
    "benign_keratosis": "bkl", "dermatofibroma": "df", "vascular_lesion": "vasc",
}


def _key(name: str) -> str:
    return re.sub(r"[\s_-]+", "_", name.strip().lower())


def canonical_label(modality: str, label: str) -> str:
    """The name this module uses for a finding label, whichever spelling the reader or vocab gave it.

    Case, spaces and underscores do not matter ("pleural thickening" is "Pleural_Thickening"), and the full
    skin names map to the dataset codes. A label this module does not know is returned unchanged."""
    vocab = VOCABS.get(modality)
    if vocab is None:
        return label
    key = _key(label)
    for name in vocab:
        if _key(name) == key:
            return name
    if modality == "skin_dermoscopy":
        return _SKIN_ALIASES.get(key, label)
    return label

_NEGATORS = re.compile(
    r"\b(?:no|without|negative for|absence of|absent|free of|denies|not|nor|resolved|clear of|"
    r"no evidence of|no signs? of|no features? of|rules? out)\b",
    re.IGNORECASE,
)
_BREAKERS = re.compile(
    r"\b(?:but|however|although|except|apart from|there (?:is|are)|with|shows?|showing|demonstrates?|noted|present|"
    r"identified|seen|reveals?|has|have|is|are)\b|[;:]",
    re.IGNORECASE,
)
_HEDGES = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|suspicious|suspected?|concerning|cannot (?:be )?(?:exclude|excluded|ruled out)|"
    r"can't (?:be )?(?:exclude|excluded|ruled out)|not (?:excluded|ruled out)|may (?:represent|be|reflect)|might|versus|vs|"
    r"question(?:able)?|equivocal|consistent with|suggest\w*|compatible with|favou?r\w*|raising|differential|could be|"
    r"cannot exclude|possible)\b",
    re.IGNORECASE,
)
_POST_NEG = re.compile(
    r"^[^.;]{0,25}?\b(?:has resolved|have resolved|resolved|is absent|are absent|not seen|not identified|not present|"
    r"is not seen|are not seen|is not identified|ruled out|excluded)\b",
    re.IGNORECASE,
)
_NORMAL = re.compile(
    r"no acute (?:cardiopulmonary|cardiac|pulmonary|osseous|intracranial|bony)\s*(?:abnormalit\w+|process|disease|findings?)|"
    r"\bnormal\s+(?:\w+\s+){0,2}(?:x-?ray|radiograph|film|study|mri|scan|examination|exam|brain|chest|lungs?)\b|"
    r"\blungs? (?:are|is) clear\b|\bbones? (?:are|is) intact\b|\bno (?:acute )?(?:focal )?abnormalit\w+|\bunremarkable\b|"
    r"\bno evidence of (?:acute )?(?:disease|abnormality)\b",
    re.IGNORECASE,
)
_NO_TUMOR = re.compile(
    r"\bno\b[^.;]{0,30}\b(?:tumou?r|mass|lesion|abnormalit\w+|enhancement)\b|\bnormal\b[^.;]{0,25}\bbrain\b|\bnormal brain\b",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"[^.!?\n]+")


@dataclass(frozen=True)
class ReportLabel:
    label: str
    present: bool
    hedged: bool
    evidence: str  # the sentence the label came from


@dataclass
class ReportLabels:
    labels: list[ReportLabel] = field(default_factory=list)
    says_normal: bool = False

    def state(self, label: str) -> str | None:
        """'present', 'hedged', 'absent' or None when the report never mentions the label."""
        for lab in self.labels:
            if lab.label == label:
                return "absent" if not lab.present else ("hedged" if lab.hedged else "present")
        return None


def _mentions(sentence: str, vocab: Vocab) -> list[tuple[str, int, int, bool]]:
    """(label, start, end, weak) for each term, longest terms first so 'actinic keratosis' beats 'keratosis'."""
    taken = [False] * len(sentence)
    out: list[tuple[str, int, int, bool]] = []
    entries = [(t, lab, weak) for lab, (strong, weak_terms) in vocab.items() for t, weak in [*[(s, False) for s in strong], *[(w, True) for w in weak_terms]]]
    entries.sort(key=lambda e: -len(e[0]))
    for term, label, weak in entries:
        for m in re.finditer(rf"\b(?:{term})\b", sentence, re.IGNORECASE):
            if any(taken[m.start() : m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            out.append((label, m.start(), m.end(), weak))
    return sorted(out, key=lambda x: x[1])


def _negated(sentence: str, start: int, end: int) -> bool:
    """True when a negator before the term has not been closed by a breaker ("but", "there is", ...)."""
    before = sentence[:start]
    last_neg = -1
    for m in _NEGATORS.finditer(before):
        if re.match(r"not\s+(?:excluded|ruled out)", before[m.start() :], re.IGNORECASE):
            continue
        last_neg = m.end()
    if last_neg >= 0 and not _BREAKERS.search(before[last_neg:]):
        return True
    return bool(_POST_NEG.match(sentence[end:]))


def _hedged(sentence: str, start: int, end: int) -> bool:
    window_before = " ".join(sentence[:start].split()[-7:])
    return bool(_HEDGES.search(window_before) or _HEDGES.search(sentence[end : end + 40]))


def labels_from_report(text: str, modality: str) -> ReportLabels:
    result = ReportLabels()
    vocab = VOCABS.get(modality)
    if not text or not text.strip() or vocab is None:
        return result
    per_label: dict[str, list[tuple[bool, bool, str]]] = {}
    for sm in _SENTENCES.finditer(text):
        sentence = sm.group(0)
        if len(sentence.strip()) < 3:
            continue
        for label, start, end, weak in _mentions(sentence, vocab):
            if _negated(sentence, start, end):
                per_label.setdefault(label, []).append((False, False, sentence.strip()))
            else:
                per_label.setdefault(label, []).append((True, weak or _hedged(sentence, start, end), sentence.strip()))
    for label, ms in per_label.items():
        positives = [m for m in ms if m[0]]
        if positives:
            hedged = all(m[1] for m in positives)
            result.labels.append(ReportLabel(label, True, hedged, positives[0][2]))
        else:
            result.labels.append(ReportLabel(label, False, False, ms[0][2]))
    any_positive = any(lab.present for lab in result.labels)
    result.says_normal = bool(_NORMAL.search(text)) and not any_positive
    if modality == "brain_mri" and not any_positive and (result.says_normal or _NO_TUMOR.search(text)):
        result.labels.append(ReportLabel("notumor", True, False, text.strip()[:200]))
    return result
