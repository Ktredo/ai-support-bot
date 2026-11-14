"""Тесты чистой пост-обработки ответов (textutils)."""
import pytest

from textutils import truncate_client_answer


def test_empty_text_returns_empty_string():
    assert truncate_client_answer("") == ""
    assert truncate_client_answer(None) == ""


def test_removes_markdown_bold_and_asterisks():
    assert truncate_client_answer("**Важно:** деньги списаны.") == "Важно: деньги списаны."


def test_removes_official_signature():
    text = "Ограничение снимется автоматически.\nС уважением, Банк"
    assert truncate_client_answer(text) == "Ограничение снимется автоматически."


def test_strips_numbered_and_bulleted_lists():
    assert truncate_client_answer("1. Первое\n2. Второе") == "Первое Второе"
    assert truncate_client_answer("- Первое\n• Второе") == "Первое Второе"


def test_joins_multiline_into_single_abzac():
    assert truncate_client_answer("Первое.\n\nВторое.\n") == "Первое. Второе."


def test_keeps_four_sentences_by_default():
    text = "Один. Два. Три. Четыре. Пять."
    assert truncate_client_answer(text) == "Один. Два. Три. Четыре."


def test_short_answer_is_not_truncated():
    text = "Деньги спишутся в рабочий день."
    assert truncate_client_answer(text) == text


def test_pizza_staff_mode_keeps_only_two_sentences():
    text = "Один. Два. Три. Четыре."
    assert truncate_client_answer(text, mode="pizza_staff") == "Один. Два."


@pytest.mark.parametrize("mode", ["bank_support", "pizza_guest", "pizza_delivery"])
def test_other_modes_use_default_limit(mode):
    text = "Один. Два. Три. Четыре. Пять."
    assert truncate_client_answer(text, mode=mode) == "Один. Два. Три. Четыре."
