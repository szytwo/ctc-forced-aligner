import re
import unicodedata

import numpy as np
from uroman import Uroman

from .norm_config import norm_config

uroman_instance = Uroman()


def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations,
    removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code : ISO 639-3 code of the language
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += r"]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + r"]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases:
    #   a)text starts with a number
    #   b) a number is present somewhere in the middle of the text
    #   c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded
    # and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:
        digits_pattern = r"[" + config["digit_set"]

        digits_pattern += r"]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        from unidecode import unidecode

        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text


# iso codes with specialized rules in uroman
special_isos_uroman = [
    "ara",
    "bel",
    "bul",
    "deu",
    "ell",
    "eng",
    "fas",
    "grc",
    "ell",
    "eng",
    "heb",
    "kaz",
    "kir",
    "lav",
    "lit",
    "mkd",
    "mkd2",
    "oss",
    "pnt",
    "pus",
    "rus",
    "srp",
    "srp2",
    "tur",
    "uig",
    "ukr",
    "yid",
]


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts: list[str], iso=None):
    outtexts = [
        uroman_instance.romanize_string(transcript, lcode=iso)
        for transcript in norm_transcripts
    ]

    uromans = []
    for ot in outtexts:
        ot = " ".join(ot.strip())
        ot = re.sub(r"\s+", " ", ot).strip()
        normalized = normalize_uroman(ot)
        uromans.append(normalized)

    assert len(uromans) == len(norm_transcripts)

    return uromans


def split_text(text: str, split_size: str = "word"):
    if split_size == "sentence":
        from nltk.tokenize import PunktSentenceTokenizer

        sentence_checker = PunktSentenceTokenizer()
        sentences = sentence_checker.sentences_from_text(text)
        return sentences

    elif split_size == "word":
        return text.split()
    elif split_size == "char":
        return list(text)


def preprocess_text(
    text, romanize, language, split_size="word", star_frequency="segment"
):
    assert split_size in [
        "sentence",
        "word",
        "char",
    ], "Split size must be sentence, word, or char"
    assert star_frequency in [
        "segment",
        "edges",
    ], "Star frequency must be segment or edges"
    if language in ["jpn", "chi"]:
        split_size = "char"
    text_split = split_text(text, split_size)
    norm_text = [text_normalize(line.strip(), language) for line in text_split]

    if romanize:
        tokens = get_uroman_tokens(norm_text, language)
    else:
        tokens = [" ".join(list(word)) for word in norm_text]

    # add <star> token to the tokens and text
    # it's used extensively here but I found that it produces more accurate results
    # and doesn't affect the runtime
    if star_frequency == "segment":
        tokens_starred = []
        [tokens_starred.extend(["<star>", token]) for token in tokens]

        text_starred = []
        [text_starred.extend(["<star>", chunk]) for chunk in text_split]

    elif star_frequency == "edges":
        tokens_starred = ["<star>"] + tokens + ["<star>"]
        text_starred = ["<star>"] + text_split + ["<star>"]

    return tokens_starred, text_starred


def normalize_segment_times(segments):
    """
    修复相邻 segment 的时间重叠。

    例如：

        A: 0.18 - 0.24
        B: 0.22 - 0.30

    修复为：

        A: 0.18 - 0.23
        B: 0.23 - 0.30

    不删除字符，只调整边界。
    """
    if len(segments) <= 1:
        return

    for i in range(len(segments) - 1):
        current = segments[i]
        next_segment = segments[i + 1]

        # 当前没有重叠
        if current["end"] <= next_segment["start"]:
            continue

        # 出现 overlap
        current["end"] = next_segment["start"]


def merge_zero_duration_segments(segments):
    """
    合并 zero-duration 字符。

    规则：
    1. zero-duration 出现在正常字符后面：
       前一个正常字符 + zero 字符 -> 合并

    2. zero-duration 连续出现：
       全部合并到前一个正常字符

    3. zero-duration 出现在开头：
       先暂存，遇到第一个正常字符后，与该字符合并

    4. 字符 text 合并
    5. 时间 start/end 同时合并
    6. 不删除任何字符
    """

    if not segments:
        return []

    results = []
    pending_zero = []

    def append_text_and_update_time(target, source):
        """
        将 source 合并到 target。
        """
        target["text"] += source["text"]
        target["start"] = min(target["start"], source["start"])
        target["end"] = max(target["end"], source["end"])

        # score 可按需要处理
        if "score" in target and "score" in source:
            target["score"] = (target["score"] + source["score"]) / 2

    for segment in segments:
        is_zero_duration = segment["end"] <= segment["start"]

        if is_zero_duration:
            # 先暂存
            pending_zero.append(segment)
            continue

        # --------------------------------------------------
        # 当前 segment 是正常字符
        # --------------------------------------------------

        current = segment.copy()

        if pending_zero:
            # --------------------------------------------------
            # 情况 A：
            # 前面已经存在正常字符
            # A  1.50 - 1.58
            # ະ  1.58 - 1.58
            # ິ  1.58 - 1.58
            # B  1.58 - 1.70
            # => Aະິ
            #    B
            # --------------------------------------------------

            if results:
                previous = results[-1]

                for zero in pending_zero:
                    append_text_and_update_time(previous, zero)

                pending_zero.clear()

                results.append(current)

            # --------------------------------------------------
            # 情况 B：
            # zero-duration 在整个结果开头
            # ະ  1.50 - 1.50
            # ິ  1.50 - 1.50
            # A  1.50 - 1.70
            # => ະິA  1.50 - 1.70
            # --------------------------------------------------

            else:
                merged = current.copy()

                # zero-duration 字符放在当前正常字符前面
                merged["text"] = (
                    "".join(zero["text"] for zero in pending_zero) + merged["text"]
                )

                # 时间范围
                merged["start"] = min(
                    [zero["start"] for zero in pending_zero] + [merged["start"]]
                )
                merged["end"] = max(
                    [zero["end"] for zero in pending_zero] + [merged["end"]]
                )

                # score
                scores = [zero["score"] for zero in pending_zero if "score" in zero]

                if "score" in merged:
                    scores.append(merged["score"])

                if scores:
                    merged["score"] = sum(scores) / len(scores)

                results.append(merged)

                pending_zero.clear()

        else:
            results.append(current)

    # ------------------------------------------------------
    # 最后还有 zero-duration
    # A  1.50 - 1.70
    # ະ  1.70 - 1.70
    # ິ  1.70 - 1.70
    # => Aະິ  1.50 - 1.70
    # ------------------------------------------------------

    if pending_zero:
        if results:
            previous = results[-1]

            for zero in pending_zero:
                append_text_and_update_time(previous, zero)
        else:
            # 整个结果都是 zero-duration
            merged = pending_zero[0].copy()

            merged["text"] = "".join(zero["text"] for zero in pending_zero)
            merged["start"] = min(zero["start"] for zero in pending_zero)
            merged["end"] = max(zero["end"] for zero in pending_zero)

            scores = [zero["score"] for zero in pending_zero if "score" in zero]

            if scores:
                merged["score"] = sum(scores) / len(scores)

            results.append(merged)

    return results


def merge_segments(segments, threshold=0.00):
    """
    合并时间间隔小于 threshold 的 segment。
    注意：
    这里是真正合并 text + start + end，
    而不是仅仅修改下一个 segment 的 start。
    """

    if not segments:
        return

    results = []

    for segment in segments:
        if not results:
            results.append(segment.copy())
            continue

        previous = results[-1]

        gap = segment["end"] - segment["start"]

        if gap < threshold:
            # 合并字符
            previous["text"] += segment["text"]
            # 合并时间
            previous["start"] = min(previous["start"], segment["start"])
            previous["end"] = max(previous["end"], segment["end"])

            # score
            if "score" in previous and "score" in segment:
                previous["score"] = (previous["score"] + segment["score"]) / 2
        else:
            results.append(segment.copy())

    segments.clear()
    segments.extend(results)


def postprocess_results(
    text_starred: list,
    spans: list,
    stride: float,
    scores: np.ndarray,
    merge_threshold: float = 0.0,
):
    results = []

    for i, t in enumerate(text_starred):
        if t == "<star>":
            continue
        span = spans[i]
        seg_start_idx = span[0].start
        seg_end_idx = span[-1].end + 1

        audio_start_sec = seg_start_idx * (stride) / 1000
        audio_end_sec = seg_end_idx * (stride) / 1000
        score = scores[seg_start_idx:seg_end_idx].mean()
        sample = {
            "start": audio_start_sec,
            "end": audio_end_sec,
            "text": t,
            "score": score.item(),
        }
        results.append(sample)

    # 处理 zero-duration 字符 + 时间一起合并
    results = merge_zero_duration_segments(results)
    # 处理 overlap
    normalize_segment_times(results)
    # 处理 threshold 字符 + 时间一起合并
    merge_segments(results, merge_threshold)
    # 处理 overlap
    normalize_segment_times(results)

    return results
