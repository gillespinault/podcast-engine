"""
Voice Selection Module
Maps detected language to appropriate Kokoro TTS voice
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Kokoro voice mapping by language and gender
# Based on Kokoro TTS 67-voice multilingual support
VOICE_MAP = {
    "fr": {
        "female": "ff_siwis",  # French female (high quality)
        "male": "fm_brune"     # French male
    },
    "en": {
        "female": "af_bella",  # American English female
        "male": "am_adam"      # American English male
    },
    "es": {
        "female": "sf_isabella",  # Spanish female
        "male": "sm_diego"        # Spanish male
    },
    "de": {
        "female": "gf_heidi",  # German female
        "male": "gm_klaus"     # German male
    },
    "it": {
        "female": "if_sofia",  # Italian female
        "male": "im_marco"     # Italian male
    },
    "pt": {
        "female": "pf_maria",  # Portuguese female
        "male": "pm_joao"      # Portuguese male
    },
    "nl": {
        "female": "nf_anna",   # Dutch female
        "male": "nm_peter"     # Dutch male
    },
    "pl": {
        "female": "plf_katarzyna",  # Polish female
        "male": "plm_jan"           # Polish male
    },
    "ru": {
        "female": "rf_natasha",  # Russian female
        "male": "rm_ivan"        # Russian male
    },
    "ja": {
        "female": "jf_yuki",  # Japanese female
        "male": "jm_takeshi"  # Japanese male
    },
    "zh": {
        "female": "zf_mei",   # Chinese female
        "male": "zm_wei"      # Chinese male
    },
    "ko": {
        "female": "kf_sora",  # Korean female
        "male": "km_minho"    # Korean male
    },
    "ar": {
        "female": "arf_leila",  # Arabic female
        "male": "arm_omar"      # Arabic male
    },
    "hi": {
        "female": "hif_priya",  # Hindi female
        "male": "him_raj"       # Hindi male
    },
    "tr": {
        "female": "trf_ayse",  # Turkish female
        "male": "trm_mehmet"   # Turkish male
    },
    "sv": {
        "female": "svf_ingrid",  # Swedish female
        "male": "svm_erik"       # Swedish male
    },
    "no": {
        "female": "nof_solveig",  # Norwegian female
        "male": "nom_lars"        # Norwegian male
    },
    "da": {
        "female": "daf_karen",  # Danish female
        "male": "dam_anders"    # Danish male
    },
    "fi": {
        "female": "fif_aino",  # Finnish female
        "male": "fim_juhani"   # Finnish male
    },
    "cs": {
        "female": "csf_marie",  # Czech female
        "male": "csm_petr"      # Czech male
    },
}

DEFAULT_VOICE = "af_bella"  # Fallback: American English female
DEFAULT_GENDER = "female"   # Default gender preference


def select_voice(
    language: str,
    gender: str = DEFAULT_GENDER,
    override: Optional[str] = None
) -> str:
    """
    Select appropriate Kokoro voice based on language

    Args:
        language: ISO 639-1 language code (fr, en, es, etc.)
        gender: "female" or "male" (default: female)
        override: Manual voice selection (bypasses auto-detection)

    Returns:
        Kokoro voice ID (e.g., "ff_siwis")

    Examples:
        >>> select_voice("fr")
        'ff_siwis'
        >>> select_voice("fr", gender="male")
        'fm_brune'
        >>> select_voice("en")
        'af_bella'
        >>> select_voice("unknown_lang")
        'af_bella'  # Fallback
        >>> select_voice("fr", override="af_bella")
        'af_bella'  # Manual override
    """
    # Manual override takes precedence
    if override:
        logger.info(f"Using manual voice override: {override}")
        return override

    # Normalize language code (handle variants like en-GB → en)
    lang_code = language.lower().split("-")[0][:2]

    # Normalize gender
    gender_normalized = gender.lower() if gender else DEFAULT_GENDER
    if gender_normalized not in ["female", "male"]:
        logger.warning(
            f"Invalid gender '{gender}', defaulting to '{DEFAULT_GENDER}'"
        )
        gender_normalized = DEFAULT_GENDER

    # Lookup voice in mapping
    if lang_code in VOICE_MAP:
        voice = VOICE_MAP[lang_code].get(
            gender_normalized,
            VOICE_MAP[lang_code].get(DEFAULT_GENDER)  # Fallback to default gender
        )
        logger.info(
            f"Voice selected: {voice} (language={lang_code}, gender={gender_normalized})"
        )
        return voice

    # Language not supported, use default
    logger.warning(
        f"No voice mapping for language '{language}', using default: {DEFAULT_VOICE}"
    )
    return DEFAULT_VOICE


def get_supported_languages() -> list[str]:
    """
    Get list of supported language codes

    Returns:
        List of ISO 639-1 language codes
    """
    return sorted(VOICE_MAP.keys())


def get_voice_info(language: str) -> Optional[Dict[str, str]]:
    """
    Get voice options for a language

    Args:
        language: ISO 639-1 language code

    Returns:
        Dictionary with female and male voice IDs, or None if not supported

    Example:
        >>> get_voice_info("fr")
        {'female': 'ff_siwis', 'male': 'fm_brune'}
    """
    lang_code = language.lower().split("-")[0][:2]
    return VOICE_MAP.get(lang_code)


def is_language_supported(language: str) -> bool:
    """
    Check if language is supported

    Args:
        language: ISO 639-1 language code

    Returns:
        True if language has dedicated voice mapping

    Example:
        >>> is_language_supported("fr")
        True
        >>> is_language_supported("swahili")
        False
    """
    lang_code = language.lower().split("-")[0][:2]
    return lang_code in VOICE_MAP
