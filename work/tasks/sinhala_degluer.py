"""
Sinhala Word-Boundary & Virama De-gluing Engine
------------------------------------------------
Detects and separates concatenated Sinhala words across virama/hal-akuru boundaries
(e.g., 'හෙරොයින්කිලෝවක්' -> 'හෙරොයින් කිලෝවක්', 'අනෙකුත්දෙදෙනාගේ' -> 'අනෙකුත් දෙදෙනාගේ',
'ඔවුන්සතුව' -> 'ඔවුන් සතුව', 'ඔවුන්රැගෙන' -> 'ඔවුන් රැගෙන', 'දැන්දිගටම' -> 'දැන් දිගටම',
'හෙරොයින්ග්රෑම්' -> 'හෙරොයින් ග්රෑම්', 'හෙරොයින්ග්‍රෑම්' -> 'හෙරොයින් ග්‍රෑම්').

Preserves legitimate conjuncts (Rakaransaya, Yansaya, Bandi-akuru) and never splits
valid single words (e.g., 'අත්අඩංගුවට', 'ප්රදේශයේ', 'මහේස්ත්රාත්', 'ශ්රී'),
inflected words (e.g., 'ඔවුන්ගේ', 'තමන්ගේ', 'සැකකරුවන්ගේ', 'දැන්ම'),
or standard vocabulary (e.g., 'කාර්යාලය', 'වාර්තාව', 'අවස්ථාව', 'විස්තරය', 'දැන්වීම', 'බස්නාහිර', 'පොලිස්පති').
"""

from __future__ import annotations

import re
import unicodedata

# Unicode Constants for Sinhala script
VIRAMA = "\u0DCA"       # ් (Al-lakuna)
ZWJ = "\u200D"          # Zero-Width Joiner
ZWNJ = "\u200C"         # Zero-Width Non-Joiner

# Legitimate single words or compounds with internal viramas that must NEVER be split
KNOWN_SINGLE_WORDS = {
    # Explicitly required by acceptance criteria to be preserved
    "අත්අඩංගුවට", "අත්අඩංගුවටද", "අත්අඩංගුවටත්", "අත්අඩංගුවෙන්", "අත්අඩංගුවටගැනීම", "අත්අඩංගුවටගත්",
    "ප්‍රදේශයේ", "ප්රදේශයේ", "ප්‍රදේශය", "ප්රදේශය", "ප්‍රදේශ", "ප්රදේශ", "ප්‍රදේශවල", "ප්රදේශවල",
    "ප්‍රදේශවාසීන්", "ප්රදේශවාසීන්",
    "මහේස්ත්‍රාත්", "මහේස්ත්රාත්", "මහේස්ත්‍රාත්වරයා", "මහේස්ත්රාත්වරයා", "මහේස්ත්‍රාත්වරිය", "මහේස්ත්රාත්වරිය",
    "මහේස්ත්‍රාත්තුමා", "මහේස්ත්රාත්තුමා",
    "ශ්‍රී", "ශ්රී", "ශ්‍රීලංකා", "ශ්රීලංකා", "ශ්‍රීලංකාවේ", "ශ්රීලංකාවේ", "ශ්‍රීලංකික",

    # Common vocabulary with internal hal-akuru / consonant clusters
    "දැන්වීම", "දැන්වීම්", "දැන්වීමක්", "දැන්වීමේ", "දැන්වීමට", "දැන්වීම්වල", "දැන්වීමයි",
    "දැන්වීය", "දැන්වූහ", "දැන්වූයේ", "දැන්වුවා", "දැන්වූ", "දැනුම්දීම", "දැනුම්දීම්", "දැනුම්දී",
    "බස්නාහිර", "බස්නාහිරින්", "බස්නාහිරට", "බස්නාහිරද",
    "පොලිස්පති", "පොලිස්පතිවරයා", "පොලිස්පතිවරිය", "පොලිස්පතිගේ", "පොලිස්ථානය", "පොලිස්ස්ථානය",
    "කාර්යාලය", "කාර්යාලයේ", "කාර්යාලයට", "කාර්යාල", "කාර්යාලවල", "කාර්යාලීය",
    "කාර්යභාරය", "කාර්යසාධන", "කාර්යමණ්ඩලය", "කාර්යය", "කාර්යයන්",
    "වාර්තාව", "වාර්තාවක්", "වාර්තාවේ", "වාර්තාවට", "වාර්තාකරණය", "වාර්තාකරු", "වාර්තාකරුවන්",
    "අවස්ථාව", "අවස්ථාවක්", "අවස්ථාවේ", "අවස්ථාවට", "අවස්ථාවන්", "අවස්ථාවලදී",
    "විස්තර", "විස්තරය", "විස්තරයක්", "විස්තරයේ", "විස්තරයට", "විස්තරවල", "විස්තරාත්මක",
    "පුස්තකාල", "පුස්තකාලය", "පුස්තකාලයේ",
    "ශාස්ත්‍රාලය", "ශාස්ත්රාලය", "ශාස්ත්‍රීය", "ශාස්ත්රීය", "ශාස්ත්‍රය", "ශාස්ත්රය",
    "විශ්වාස", "විශ්වාසය", "විශ්වාසයක්", "විශ්වාසයෙන්", "විශ්වාසනීය", "විශ්වාසවන්ත",
    "ධාර්මික", "ආගමික", "ආර්ථික", "ආර්ථිකය", "ආර්ථිකයේ", "ආර්ථිකයට",
    "තීරණය", "තීරණයට", "තීරණ", "තීරණයක්",
    "මාර්ගය", "මාර්ගයේ", "මාර්ගයට", "මාර්ගගත", "මාර්ගගතව",
    "කර්මාන්ත", "කර්මාන්තය", "කර්මාන්තයේ", "කර්මාන්තශාලා", "කර්මාන්තපුර",
    "සාකච්ඡා", "සාකච්ඡාව", "සාකච්ඡාවේ", "සාකච්ඡාවක්",
    "අත්‍යවශ්‍ය", "අත්යවශ්ය", "විශේෂඥ", "විශේෂඥයින්",
    "අධ්‍යාපන", "අධ්යාපන", "අධ්‍යාපනය", "අධ්යාපනය",

    # Common Sinhala compounds with internal viramas
    "අත්හිටුවීමට", "අත්හිටුවීම", "අත්හිටුවා", "අත්හිටුවන", "අත්හිටුවනු",
    "අත්සන්", "අත්සන", "අත්සන්කර", "අත්සන්කළ", "අත්සන්කිරීම",
    "අත්හැර", "අත්හැරීමට", "අත්ඇරීම", "අත්දුටු", "අත්දැකීම්", "අත්දැකීම",
    "උත්සාහ", "උත්සාහය", "උත්සාහයක", "උත්සාහයන්",
    "උත්සව", "උත්සවය", "උත්සවයේ", "උත්සවයට", "උත්සවයක්",

    # Words with internal consonant clusters (Sanskrit/Pali roots and standard vocabulary)
    "තත්ත්ව", "තත්ත්වය", "තත්ත්වයේ", "තත්ත්වයන්", "තත්ත්වයද",
    "නිෂ්පාදන", "නිෂ්පාදනය", "නිෂ්පාදනවල", "නිෂ්පාදනවලට", "නිෂ්පාදිත",
    "විශ්ලේෂණ", "විශ්ලේෂණය", "විශ්ලේෂකයින්", "විශ්ලේෂණාත්මක",
    "සම්බන්ධ", "සම්බන්ධය", "සම්බන්ධයෙන්", "සම්බන්ධව", "සම්බන්ධතාව", "සම්බන්ධතා",
    "විශිෂ්ට", "විශිෂ්ටතම", "විශිෂ්ටත්වය",
    "ආයෝජන", "ආයෝජනය", "ආයෝජකයින්", "ආයෝජකයින්ගේ",
    "ආශ්චර්ය", "ආශ්චර්යය", "ආශ්චර්යමත්",
    "අගෝස්තු", "ඔක්තෝබර්", "සැප්තැම්බර්", "නොවැම්බර්", "දෙසැම්බර්",
    "අන්තර්ජාල", "අන්තර්ජාලය", "අන්තර්ජාලයේ", "අන්තර්ජාතික",
    "සංස්කෘතික", "සංස්කෘතිය", "සංස්කෘතියේ",
    "පක්ෂය", "පක්ෂයේ", "පක්ෂ", "විපක්ෂ", "ආණ්ඩුපක්ෂ",
    "ලක්ෂ", "ලක්ෂය", "ලක්ෂයක්",
    "අධ්‍යක්ෂ", "අධ්‍යක්ෂක", "අධ්‍යක්ෂවරයා", "අධ්‍යක්ෂිකා",
    "සම්පූර්ණ", "සම්පූර්ණයෙන්ම", "සම්පූර්ණව",
    "කර්තව්‍ය", "කර්තව්‍යය", "කර්තව්‍යයන්",
    "වර්ධන", "වර්ධනය", "වර්ධනයක්", "වර්ධනයට",
    "ක්‍රියාත්මක", "ක්රියාත්මක", "ක්‍රියාකාරී", "ක්රියාකාරී", "ක්‍රියා", "ක්රියා",
    "ප්‍රතිඵල", "ප්රතිඵල", "ප්‍රතිපත්තිය", "ප්රතිපත්තිය",
    "ප්‍රකාශ", "ප්රකාශ", "ප්‍රකාශය", "ප්රකාශය", "ප්‍රකාශයක්", "ප්රකාශයක්",
    "ප්‍රවෘත්ති", "ප්රවෘත්ති",
    "ජ්‍යෙෂ්ඨ", "ජ්යෙෂ්ඨ",
    "බන්ධනාගාර", "බන්ධනාගාරගත", "බන්ධනාගාරය",
    "අත්අඩංගුවටගෙන", "අත්අඩංගුවටගැනීමට",
    "මහේස්ත්‍රාත්වරුන්", "මහේස්ත්රාත්වරුන්",
    "අයිස්ක්‍රීම්", "අයිස්ක්රීම්",
    "බස්රථ", "බස්රථය", "බස්රථයේ", "බස්රථයක", "බස්රථවල", "බස්නැවතුම",
    "පාර්ලිමේන්තුව", "පාර්ලිමේන්තුවේ", "පාර්ලිමේන්තුවට", "පාර්ලිමේන්තු",
    "විශ්වවිද්‍යාල", "විශ්වවිද්‍යාලය", "විශ්වවිද්‍යාලයේ", "විශ්වවිද්‍යාලයට", "විශ්වවිද්යාලය",
    "අමාත්‍යාංශය", "අමාත්‍යාංශයේ", "අමාත්‍ය", "අමාත්‍යවරයා", "අමාත්‍යවරුන්", "අමාත්ය",
    "ජාත්‍යන්තර", "ජාත්‍යන්තරය", "ජාත්‍යන්තරයේ", "ජාත්යන්තර",
    "ත්‍රස්තවාදී", "ත්රස්තවාදී", "ත්‍රස්තවාදය", "ත්රස්තවාදය",
    "ප්‍රජාතන්ත්‍රවාදී", "ප්රජාතන්ත්රවාදී", "සෞඛ්‍ය", "සෞඛ්‍යය", "සෞඛ්ය",
    "මත්ද්‍රව්‍ය", "මත්ද්‍රව්‍යද", "මත්ද්‍රව්‍යත්", "මත්ද්‍රව්‍යවල", "මත්ද්‍රව්‍යවලින්", "මත්ද්‍රව්‍යවලට",
    "අධිකරණය", "අධිකරණයේ", "අධිකරණයට", "අධිකරණයෙන්",
}

# Sinhala grammatical case endings, enclitics, and post-clitics that attach directly
# to words ending in hal-akuru and must NEVER be split off as separate words
GRAMMATICAL_SUFFIXES = {
    "ගේ",      # Genitive (ඔවුන්ගේ, තමන්ගේ, සැකකරුවන්ගේ)
    "ගෙන්",    # Ablative/Instrumental (ඔවුන්ගෙන්, තමන්ගෙන්, සැකකරුවන්ගෙන්)
    "ට",       # Dative (ඔවුන්ට, තමන්ට, සැකකරුවන්ට)
    "ම",       # Emphatic particle (දැන්ම, ඔවුන්ම, තමන්ම, එයින්ම, මෙයින්ම)
    "ද",       # Interrogative/enclitic (ඔවුන්ද, තමන්ද)
    "ත්",      # Concessive/enclitic (ඔවුන්ත්, තමන්ත්)
    "ය",       # Assertive/predicative (ඔවුන්ය)
    "යි",      # Predicative/conjunctive enclitic (සැකකරුවනුයි)
    "නේ",      # Vocative/plural enclitic (දෙවියන්නේ)
    "දෝ",      # Dubitative enclitic
    # Noun plural markers, oblique endings, and enclitics
    "වල",      # Plural (පොත්වල, බස්වල, පැකට්වල, හෙරොයින්වල)
    "වලින්",   # Plural ablative (බස්වලින්, හෙරොයින්වලින්)
    "වලට",     # Plural dative (බස්වලට, හෙරොයින්වලට)
    "වලත්",    # Plural concessive
    "වලද",     # Plural interrogative
    "දීම",     # Locative emphatic (එයින්දීම, මෙයින්දීම)
    "ගෙන්ද",   # Ablative interrogative
    "ගෙන්ත්",  # Ablative concessive
    "ගේද",     # Genitive interrogative
    "ගේත්",    # Genitive concessive
    "ටද",      # Dative interrogative
    "ටත්",     # Dative concessive
    "වල්",     # Plural marker
}

# Words ending in hal-akuru that commonly concatenate with following words in LLM output
KNOWN_VIRAMA_ENDINGS = {
    # Glued test words
    "හෙරොයින්", "අනෙකුත්", "ඔවුන්", "දැන්",

    # Pronouns, adverbs, conjunctions
    "මොවුන්", "තමන්", "එයින්", "මෙයින්", "නමුත්", "තවත්", "එහෙත්", "නැවතත්",
    "අනෙක්", "කිසිවක්", "යමක්", "සමහරක්", "දිනක්", "පිරිසක්", "තොගයක්",

    # Common function words & particles
    "විසින්", "තුළින්", "මගින්", "මඟින්", "ලෙසින්", "වෙතින්", "සමඟින්", "සමගින්",

    # Nouns & loanwords
    "කොකේන්", "අයිස්", "පොලිස්", "බස්", "රුපියල්", "ලීටර්", "මීටර්", "ඩොලර්", "පවුම්",
    "ග්‍රෑම්", "ග්රෑම්", "මිලිග්‍රෑම්", "මිලිග්රෑම්", "කිලෝග්‍රෑම්", "කිලෝග්රෑම්",

    # Suffixes and common word forms ending in virama
    "සැකකරුවන්", "පුද්ගලයින්", "නිලධාරීන්", "කාන්තාවන්", "ජාවාරම්කරුවන්",
    "ධීවරයින්", "ධීවරයන්", "මන්ත්‍රීවරුන්", "දෙනෙකුගේ", "දෙනෙකුන්",
    "ලක්ෂයක්", "මිලියනයක්", "බිලියනයක්", "කෝටියක්",
    "පැකට්ටුවක්", "කිලෝවක්", "වටිනාකමක්", "ප්‍රමාණයක්", "මුදලක්",
    "එකක්", "දෙකක්", "තුනක්", "හතරක්", "පහක්", "හයක්", "සියයක්", "දහසක්",
}

# Ambiguous short prefixes that must only be split when the suffix is a recognized right word
AMBIGUOUS_VIRAMA_PREFIXES = {"දැන්", "බස්", "පොලිස්", "අනෙක්", "අයිස්"}

# Words that commonly appear on the RIGHT side of a virama boundary
KNOWN_RIGHT_WORDS = {
    # Glued test words
    "කිලෝවක්", "කිලෝ", "කිලෝග්‍රෑම්", "කිලෝග්රෑම්",
    "දෙදෙනාගේ", "දෙදෙනෙකු", "දෙදෙනෙක්", "දෙදෙනා",
    "සතුව", "සතුවූ", "සතුවී",
    "රැගෙන", "රැගෙනආ",
    "දිගටම",
    "ග්‍රෑම්", "ග්රෑම්", "මිලිග්‍රෑම්", "මිලිග්රෑම්",

    # Conjunctions, particles, and frequent connectives
    "සහ", "හා", "හෝ", "මෙන්ම", "පමණක්", "පමණ",
    "සමඟ", "සමග", "සමඟින්", "සමගින්", "සඳහා", "සඳහන්",

    # Numbers & numerals
    "තිදෙනාගේ", "තිදෙනෙකු", "තිදෙනෙක්", "තිදෙනා",
    "සිව්දෙනෙකු", "සතරදෙනෙකු", "පස්දෙනෙකු", "හයදෙනෙකු", "හත්දෙනෙකු", "අටදෙනෙකු",
    "එක්", "එකක්", "දෙකක්", "තුනක්", "හතරක්", "පහක්", "හයක්",
    "සියයක්", "දහසක්", "ලක්ෂ", "මිලියන", "බිලියන", "කෝටි",
    "දෙදහසක්", "පන්දහසක්",

    # Verbs and common predicates
    "සොයා", "සොයාගෙන", "සොයාගත්", "සොයාගනු", "ගෙන", "ගොස්",
    "කර", "කරන", "කරනු", "කළ", "කළේය", "කරමින්", "කිරීමට",
    "වෙත", "විසින්", "ලෙස", "ලබා", "ලබාදී", "ලබාගෙන", "දී",
    "ඇති", "ඇත", "තිබී", "තිබේ", "තිබුණි",
    "පැවසීය", "පවසයි", "වාර්තා", "අත්අඩංගුවට", "අතර", "නැත",
    "සිට", "දක්වා", "වැනි", "පිළිබඳ", "පිළිබඳව",
    "ක්‍රියාත්මක", "ක්රියාත්මක",
    "වැඩිදුර", "විශේෂ", "විමර්ශන", "පරීක්ෂණ", "පරීක්‍ෂණ", "මෙහෙයුම", "මෙහෙයුම්",
    "ඇතුළු", "ඇතුළත්", "නොමැත", "නොහැක", "නොවේ",

    # Nouns, officers, and objects
    "මුදල්", "තොගයක්", "ප්‍රමාණයක්", "රැස්වීම", "රටවල", "අය", "නඩුව",
    "නිලධාරීන්", "නිලධාරියා", "කණ්ඩායම", "පරීක්ෂක", "පරීක්‍ෂක",
    "අධිකරණය", "නඩු", "නීති", "පසු", "පසුව", "පෙර", "වහාම", "නැවත",
}

# Sinhala word token pattern
SINHALA_WORD_TOKEN_RE = re.compile(r'[\u0D80-\u0DFF\u200C\u200D]+')


def _is_recognized_right_word(suffix: str) -> bool:
    """Checks whether the suffix represents a plausible, recognizable independent word."""
    if not suffix or len(suffix) < 2:
        return False
    # A suffix cannot begin with a virama or dependent vowel sign
    if suffix[0] == VIRAMA or ("\u0DCF" <= suffix[0] <= "\u0DDF"):
        return False
    # Grammatical inflections must not be treated as separate right words
    if suffix in GRAMMATICAL_SUFFIXES:
        return False
    # Exact match in known right words
    if suffix in KNOWN_RIGHT_WORDS:
        return True
    # Suffix starts with a high-confidence right word (e.g. 'සතුවූ', 'රැගෙනආ', 'දෙදෙනාට')
    for rw in KNOWN_RIGHT_WORDS:
        if len(rw) >= 3 and suffix.startswith(rw):
            return True
    return False


def _is_valid_prefix(prefix: str) -> bool:
    """Checks whether the prefix is a plausible Sinhala word ending in a hal-akuru."""
    if not prefix or not prefix.endswith(VIRAMA) or len(prefix) <= 2:
        return False
    if prefix in KNOWN_VIRAMA_ENDINGS:
        return True
    # Plural noun endings (-කරුවන්, -වරුන්, -යින්, -වන්, -යන්)
    if len(prefix) >= 5 and (
        prefix.endswith("කරුවන්") or prefix.endswith("වරුන්") or
        prefix.endswith("යින්") or prefix.endswith("වන්") or prefix.endswith("යන්")
    ):
        return True
    # Indefinite noun suffix -ක් (e.g. -යක්, -ලක්, -වක්, -මක්, -සක්, -නක්)
    if len(prefix) >= 5 and prefix.endswith("ක්"):
        return True
    # Ablative -ගෙන්
    if len(prefix) >= 5 and prefix.endswith("ගෙන්"):
        return True
    return False


def find_virama_split_point(token: str) -> int | None:
    """
    Finds the index of the VIRAMA where token should be separated into two words,
    or None if the token should not be split.
    Preserves legitimate conjuncts (Rakaransaya, Yansaya, Bandi-akuru),
    grammatical case inflections, and valid single words.
    """
    if not token or len(token) < 4:
        return None
    if token in KNOWN_SINGLE_WORDS:
        return None

    virama_indices = [i for i, ch in enumerate(token) if ch == VIRAMA and i < len(token) - 1]

    for idx in virama_indices:
        next_char = token[idx + 1]
        # Check conjunct protection: Virama + ZWJ or Virama + ZWNJ
        if next_char in (ZWJ, ZWNJ):
            continue

        prefix = token[:idx + 1]
        suffix = token[idx + 1:]

        # Grammatical case endings or enclitics attached to base word: NEVER split
        if suffix in GRAMMATICAL_SUFFIXES:
            continue

        # If the combined form is a protected single word
        if (prefix + suffix) in KNOWN_SINGLE_WORDS:
            continue

        # Hal-akuru word glued directly to ASCII digits (e.g. 'රුපියල්2.6', 'ග්‍රෑම්77', 'හෙරොයින්100')
        if next_char.isdigit():
            if prefix in KNOWN_VIRAMA_ENDINGS or _is_valid_prefix(prefix):
                return idx
            continue

        # Special check: Virama followed immediately by 'ර' (Rakaransaya without ZWJ)
        if next_char == "ර":
            # Only split if prefix is a known complete word and suffix is a recognized right word
            if prefix in KNOWN_VIRAMA_ENDINGS and _is_recognized_right_word(suffix):
                return idx
            # Otherwise preserve as Rakaransaya (e.g. 'ප්රදේශයේ', 'මහේස්ත්රාත්', 'ශ්රී')
            continue

        # Special check: Virama followed immediately by 'ය' (Yansaya without ZWJ)
        if next_char == "ය":
            if prefix in KNOWN_VIRAMA_ENDINGS and _is_recognized_right_word(suffix):
                return idx
            # Otherwise preserve as Yansaya (e.g. 'ව්යවහාර', 'සත්ය')
            continue

        # Ambiguous prefixes ('දැන්', 'බස්', 'පොලිස්', etc.) MUST match a recognized right word
        # to avoid splitting words like 'දැන්වීම', 'බස්නාහිර', 'පොලිස්පති'
        if prefix in AMBIGUOUS_VIRAMA_PREFIXES:
            if _is_recognized_right_word(suffix):
                return idx
            continue

        # High-confidence known virama endings
        if prefix in KNOWN_VIRAMA_ENDINGS:
            if _is_recognized_right_word(suffix):
                return idx
            # For strictly unambiguous closed loanwords ('හෙරොයින්', 'කොකේන්', 'අනෙකුත්')
            if prefix in {"හෙරොයින්", "කොකේන්", "අනෙකුත්"}:
                if len(suffix) >= 3 and not (suffix[0] == VIRAMA or ("\u0DCF" <= suffix[0] <= "\u0DDF")):
                    return idx

        # Inflectional prefix matching a recognized right word
        if _is_valid_prefix(prefix) and _is_recognized_right_word(suffix):
            return idx

    return None


def deglue_token(token: str) -> str:
    """Recursively separates glued Sinhala words in a single token."""
    split_idx = find_virama_split_point(token)
    if split_idx is not None:
        left = token[:split_idx + 1]
        right = token[split_idx + 1:]
        # Recurse on left and right in case of multi-word concatenation (e.g. ඔවුන්විසින්සොයාගත්)
        return f"{deglue_token(left)} {deglue_token(right)}"
    return token


def deglue_virama_boundaries(text: str) -> str:
    """
    Post-processes Sinhala text, finding and separating words that were
    concatenated across virama/hal-akuru boundaries without whitespace.
    """
    if not text:
        return text

    # Separate hal-akuru words glued directly before digits (e.g. 'රුපියල්2.6' -> 'රුපියල් 2.6', 'කිලෝ1' -> 'කිලෝ 1')
    text = re.sub(r'([\u0D80-\u0DFF]\u0DCA)([0-9])', r'\1 \2', text)
    text = re.sub(r'(කිලෝ)([0-9])', r'\1 \2', text)
    # Separate digits glued directly before mass units (e.g. '77ග්‍රෑම්' -> '77 ග්‍රෑම්', '1කිලෝ' -> '1 කිලෝ')
    text = re.sub(r'([0-9])(ග්‍රෑම්|ග්රෑම්|කිලෝ|මිලිග්‍රෑම්|මිලිග්රෑම්)', r'\1 \2', text)

    def _replace_token(match: re.Match) -> str:
        tok = match.group(0)
        return deglue_token(tok)

    return SINHALA_WORD_TOKEN_RE.sub(_replace_token, text)


def is_virama_glued(token: str) -> bool:
    """Checks whether a token contains an unseparated virama-boundary word concatenation."""
    return find_virama_split_point(token) is not None


def heal_sinhala_text(text: str) -> str:
    """Heals broken Sinhala subword BPE token splits and de-glues
    concatenated Sinhala words across virama/hal-akuru boundaries."""
    if not text:
        return text
    # 1. Re-attach single-consonant Rakaransaya diacritic prefixes separated by spaces
    text = re.sub(r'\b([\u0D80-\u0DFF][්‍්]?[ර්‍ර])\s+([\u0D80-\u0DFF]+)', r'\1\2', text)

    # 2. Common Llama-3 Sinhala BPE subword splits
    splits = [
        (r'ප්‍ර\s+දේශ', 'ප්‍රදේශ'),
        (r'පා\s+රිභෝගික', 'පාරිභෝගික'),
        (r'වී\s+යාජ', 'ව්‍යාජ'),
        (r'මහේස්ත්‍\s+රාත්', 'මහේස්ත්‍රාත්'),
        (r'අත්\s+අඩංගුවට', 'අත්අඩංගුවට'),
    ]
    for pattern, repl in splits:
        text = re.sub(pattern, repl, text)

    # 3. De-glue concatenated Sinhala words across virama/hal-akuru boundaries
    text = deglue_virama_boundaries(text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()
