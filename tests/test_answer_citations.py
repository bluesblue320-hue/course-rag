"""Tests for the strict ``[来源N]`` citation parser."""

import pytest

from src.evaluation.answer_citations import (
    citation_numbers,
    parse_citations,
    unique_cited_sources,
)


class TestParseCitations:
    def test_single_valid_citation(self) -> None:
        result = parse_citations("根据资料回答。[来源1]")
        assert len(result.valid_syntax) == 1
        occurrence = result.valid_syntax[0]
        assert occurrence.source_number == 1
        assert result.malformed_fragments == ()

    def test_multiple_consecutive_citations(self) -> None:
        result = parse_citations("事实一[来源1][来源2]事实二")
        assert [o.source_number for o in result.valid_syntax] == [1, 2]

    def test_repeated_citations_are_kept_separately(self) -> None:
        result = parse_citations("事实[来源2]补充[来源2]")
        assert [o.source_number for o in result.valid_syntax] == [2, 2]
        assert len(result.valid_syntax) == 2

    def test_two_digit_citation(self) -> None:
        result = parse_citations("说明[来源12]")
        assert len(result.valid_syntax) == 1
        assert result.valid_syntax[0].source_number == 12

    def test_zero_citation_is_malformed(self) -> None:
        result = parse_citations("说明[来源0]")
        assert result.valid_syntax == ()
        assert "[来源0]" in result.malformed_fragments

    def test_leading_zero_citation_is_malformed(self) -> None:
        result = parse_citations("说明[来源01]")
        assert result.valid_syntax == ()
        assert "[来源01]" in result.malformed_fragments

    def test_empty_brackets_are_malformed(self) -> None:
        result = parse_citations("说明[来源]")
        assert result.valid_syntax == ()
        assert "[来源]" in result.malformed_fragments

    def test_letter_citation_is_malformed(self) -> None:
        result = parse_citations("说明[来源A]")
        assert result.valid_syntax == ()
        assert "[来源A]" in result.malformed_fragments

    def test_spaced_citation_is_malformed(self) -> None:
        result = parse_citations("说明[来源 1]")
        assert result.valid_syntax == ()
        assert "[来源 1]" in result.malformed_fragments

    def test_full_width_bracket_is_malformed(self) -> None:
        result = parse_citations("说明【来源1】")
        assert result.valid_syntax == ()
        assert "【来源1】" in result.malformed_fragments

    def test_comma_separated_citation_is_malformed(self) -> None:
        result = parse_citations("说明[来源1, 来源2]")
        assert result.valid_syntax == ()
        assert "[来源1, 来源2]" in result.malformed_fragments

    def test_out_of_range_number_is_still_valid_syntax(self) -> None:
        # Syntax validity and source-range validity are separate concerns:
        # the parser only checks the format, the metrics layer checks range.
        result = parse_citations("说明[来源99]")
        assert len(result.valid_syntax) == 1
        assert result.valid_syntax[0].source_number == 99

    def test_no_citations(self) -> None:
        result = parse_citations("没有任何引用格式的答案。")
        assert result.valid_syntax == ()
        assert result.malformed_fragments == ()

    def test_empty_answer(self) -> None:
        result = parse_citations("")
        assert result.valid_syntax == ()
        assert result.malformed_fragments == ()

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            parse_citations(123)  # type: ignore[arg-type]


class TestCitationHelpers:
    def test_citation_numbers_preserve_order(self) -> None:
        result = parse_citations("一[来源3]二[来源1]三[来源3]")
        assert citation_numbers(result.valid_syntax) == (3, 1, 3)

    def test_unique_cited_sources_keep_first_appearance(self) -> None:
        result = parse_citations("一[来源3]二[来源1]三[来源3]")
        assert unique_cited_sources(result.valid_syntax) == (3, 1)


class TestCitationsDoNotInterfereWithFactMatching:
    def test_citation_text_is_not_part_of_fact_phrases(self) -> None:
        from src.evaluation.answer_metrics import normalize_answer_text

        # The citation marker itself is normalized text, but fact phrases are
        # matched against the whole answer; a phrase containing the citation
        # marker must not be silently invented.
        normalized = normalize_answer_text("答案是事实。[来源1]")
        assert "来源1" in normalized
        # The normalized form keeps the citation marker inside the answer, so
        # a fact phrase that happens to include the marker could match; the
        # parser itself never strips or rewrites answer text.
        assert "事实.[来源1]" in normalized
