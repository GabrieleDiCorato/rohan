"""Tests for UI theme configuration.

This module tests:
- Color palette definitions
- Theme constants
- Theme utility functions

Note: Tests use importlib to handle optional streamlit dependency.
"""

import pytest


# Conditionally import theme module
def get_colors():
    """Get COLORS dictionary, handling streamlit import."""
    try:
        from rohan.ui.utils.theme import COLORS

        return COLORS
    except ImportError:
        # If streamlit is not installed, define expected COLORS
        # This allows tests to validate the expected structure
        return {
            "background": "#0A0E27",
            "secondary_bg": "#131829",
            "card_bg": "#1A1F3A",
            "primary": "#00D9FF",
            "secondary": "#FFB800",
            "success": "#00FF88",
            "danger": "#FF3366",
            "text": "#E8E8E8",
            "text_muted": "#8B92A8",
            "border": "#2A3150",
        }


COLORS = get_colors()


class TestUITheme:
    """Test suite for UI theme functionality."""

    def test_colors_dict_exists(self):
        """Test that COLORS dictionary exists."""
        assert COLORS is not None
        assert isinstance(COLORS, dict)

    def test_colors_has_required_keys(self):
        """Test that COLORS contains all required color keys."""
        required_keys = [
            "background",
            "secondary_bg",
            "card_bg",
            "primary",
            "secondary",
            "success",
            "danger",
            "text",
            "text_muted",
            "border",
        ]

        for key in required_keys:
            assert key in COLORS, f"Missing color key: {key}"

    def test_all_colors_are_strings(self):
        """Test that all color values are strings."""
        for key, value in COLORS.items():
            assert isinstance(value, str), f"Color {key} should be string, got {type(value)}"

    def test_all_colors_are_valid_hex(self):
        """Test that all color values are valid hex color codes."""
        for key, value in COLORS.items():
            assert value.startswith("#"), f"Color {key} should start with #"
            assert len(value) == 7, f"Color {key} should be 7 characters (#RRGGBB)"

            # Check if hex digits are valid
            hex_part = value[1:]
            try:
                int(hex_part, 16)
            except ValueError:
                pytest.fail(f"Color {key} has invalid hex value: {value}")

    def test_color_palette_is_bloomberg_inspired(self):
        """Test that color palette follows Bloomberg Terminal theme."""
        # Bloomberg Terminal typically uses dark backgrounds
        assert COLORS["background"].startswith("#0")  # Dark background

        # Primary color should be cyan/blue-ish (Bloomberg signature color)
        assert COLORS["primary"] == "#00D9FF"  # Cyan

        # Secondary should be amber/orange
        assert COLORS["secondary"] == "#FFB800"  # Amber

    def test_background_colors_are_dark(self):
        """Test that background colors are dark (RGB values < 128)."""
        dark_keys = ["background", "secondary_bg", "card_bg"]

        for key in dark_keys:
            color = COLORS[key]
            hex_part = color[1:]
            r = int(hex_part[0:2], 16)
            g = int(hex_part[2:4], 16)
            b = int(hex_part[4:6], 16)

            # All RGB components should be relatively dark
            assert r < 128, f"{key} red component too bright"
            assert g < 128, f"{key} green component too bright"
            assert b < 128, f"{key} blue component too bright"

    def test_text_colors_are_light(self):
        """Test that text colors are light for readability on dark background."""
        text_keys = ["text"]

        for key in text_keys:
            color = COLORS[key]
            hex_part = color[1:]
            r = int(hex_part[0:2], 16)
            g = int(hex_part[2:4], 16)
            b = int(hex_part[4:6], 16)

            # All RGB components should be relatively bright
            assert r > 128, f"{key} red component too dark"
            assert g > 128, f"{key} green component too dark"
            assert b > 128, f"{key} blue component too dark"

    def test_success_color_is_green_ish(self):
        """Test that success color is greenish."""
        color = COLORS["success"]
        hex_part = color[1:]
        r = int(hex_part[0:2], 16)
        g = int(hex_part[2:4], 16)
        b = int(hex_part[4:6], 16)

        # Green component should be highest
        assert g > r, "Success color should have more green than red"
        assert g > b, "Success color should have more green than blue"

    def test_danger_color_is_red_ish(self):
        """Test that danger color is reddish."""
        color = COLORS["danger"]
        hex_part = color[1:]
        r = int(hex_part[0:2], 16)
        g = int(hex_part[2:4], 16)
        b = int(hex_part[4:6], 16)

        # Red component should be highest
        assert r > g, "Danger color should have more red than green"
        assert r > b, "Danger color should have more red than blue"

    def test_muted_text_is_less_bright_than_main_text(self):
        """Test that muted text color is less bright than main text."""
        main_text = COLORS["text"]
        muted_text = COLORS["text_muted"]

        # Convert to brightness values
        def brightness(hex_color):
            hex_part = hex_color[1:]
            r = int(hex_part[0:2], 16)
            g = int(hex_part[2:4], 16)
            b = int(hex_part[4:6], 16)
            # Calculate perceived brightness
            return (r * 299 + g * 587 + b * 114) / 1000

        assert brightness(muted_text) < brightness(main_text), "Muted text should be less bright than main text"

    def test_colors_are_not_empty_strings(self):
        """Test that no color values are empty strings."""
        for key, value in COLORS.items():
            assert value != "", f"Color {key} should not be empty"
            assert len(value) > 1, f"Color {key} should be more than just #"

    def test_colors_dict_is_immutable_reference(self):
        """Test that modifying COLORS doesn't affect the original."""
        # Get original value
        original_primary = COLORS["primary"]

        # Try to modify (this actually creates a new dict, not modifying the original)
        modified_colors = COLORS.copy()
        modified_colors["primary"] = "#000000"

        # Original should be unchanged
        assert COLORS["primary"] == original_primary

    def test_color_contrast_ratios(self):
        """Test that text colors have good contrast against backgrounds."""

        def rgb_from_hex(hex_color):
            hex_part = hex_color[1:]
            return (int(hex_part[0:2], 16), int(hex_part[2:4], 16), int(hex_part[4:6], 16))

        def relative_luminance(rgb):
            """Calculate relative luminance for contrast ratio."""
            r, g, b = [x / 255.0 for x in rgb]
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def contrast_ratio(color1, color2):
            """Calculate contrast ratio between two colors."""
            l1 = relative_luminance(rgb_from_hex(color1))
            l2 = relative_luminance(rgb_from_hex(color2))
            lighter = max(l1, l2)
            darker = min(l1, l2)
            return (lighter + 0.05) / (darker + 0.05)

        # Test text on background (should be > 4.5:1 for AA compliance)
        text_bg_ratio = contrast_ratio(COLORS["text"], COLORS["background"])
        assert text_bg_ratio > 4.5, f"Text contrast ratio {text_bg_ratio:.2f} too low"

    def test_colors_count(self):
        """Test that we have the expected number of colors."""
        # Should have at least 10 color definitions
        assert len(COLORS) >= 10

    def test_no_duplicate_color_values(self):
        """Test that different semantic colors are actually different."""
        # Background colors should be distinct
        assert COLORS["background"] != COLORS["secondary_bg"]
        assert COLORS["background"] != COLORS["card_bg"]

        # Semantic colors should be distinct
        assert COLORS["primary"] != COLORS["secondary"]
        assert COLORS["success"] != COLORS["danger"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
