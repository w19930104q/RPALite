"""Fully offline synthetic-image regression tests for multi-scale image locating.

No screenshots, OCR models, network access or display are involved: every test
image is generated in memory with PIL/numpy. Coverage:

* multi-scale search (template found only at non-1.0 scales, e.g. 150%/200%)
* non-maximum suppression of overlapping candidates (one control -> one hit)
* correlation threshold boundary (matches at the threshold, none just above it)
* independent dual-axis scaling (different x/y scale factors)
* the same icon appearing multiple times (each occurrence returns once)
* deterministic behavior across consecutive calls
* edge cases: empty images, templates larger than the parent, None inputs
* preservation of the legacy single-hit / all-hit APIs
* RPALite-level physical -> logical coordinate conversion
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from RPALite import ImageHandler


# ---------------------------------------------------------------------------
# Deterministic synthetic image helpers
# ---------------------------------------------------------------------------

def make_template(size=(40, 40), seed=7):
    """A distinctive, non-uniform template (solid backgrounds would make
    TM_CCOEFF_NORMED undefined)."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    # High-contrast geometric markers give the template a strong signature.
    draw.rectangle([2, 2, size[0] - 3, size[1] - 3], outline=(255, 0, 0), width=3)
    draw.ellipse([8, 8, size[0] - 9, size[1] - 9], outline=(0, 255, 0), width=2)
    draw.line([4, size[1] - 5, size[0] - 5, 4], fill=(0, 0, 255), width=2)
    return img


def make_canvas(size=(320, 240), seed=21):
    """A noisy canvas so background regions never correlate perfectly."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(30, 200, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(noise, mode="RGB")


def paste(canvas, template, pos):
    canvas = canvas.copy()
    canvas.paste(template, pos)
    return canvas


def center_of(rect):
    x, y, w, h = rect[:4]
    return x + w // 2, y + h // 2


@pytest.fixture(scope="class")
def handler():
    # Lazy OCR initialization means no OCR backend is needed for these tests.
    return ImageHandler(debug_mode=False)


class TestMultiScaleSearch:
    def test_finds_template_at_identity_scale(self, handler):
        template = make_template()
        pos = (60, 50)
        canvas = paste(make_canvas(), template, pos)

        matches = handler.find_image_matches(template, canvas, scales=(1.0,))
        assert len(matches) == 1
        x, y, w, h, score = matches[0]
        assert (x, y) == pos
        assert (w, h) == template.size
        assert 0.0 <= score <= 1.0
        assert score >= 0.9

    def test_finds_template_scaled_up_150_and_200_percent(self, handler):
        template = make_template()
        canvas = make_canvas(size=(420, 340))
        placements = []
        for i, scale in enumerate((1.5, 2.0)):
            scaled = template.resize(
                (round(template.size[0] * scale), round(template.size[1] * scale)),
                Image.LANCZOS,
            )
            pos = (20 + i * 200, 30 + i * 40)
            canvas = paste(canvas, scaled, pos)
            placements.append((pos, scaled.size))

        # At scale 1.0 alone nothing should be found; the multi-scale search
        # must locate both enlarged occurrences.
        assert handler.find_image_matches(template, canvas, scales=(1.0,)) == []

        matches = handler.find_image_matches(
            template, canvas, scales=(1.0, 1.25, 1.5, 1.75, 2.0)
        )
        assert len(matches) == 2
        for (expected_pos, expected_size), match in zip(placements, sorted(matches, key=lambda m: m[0])):
            x, y, w, h, score = match
            # Resizing introduces interpolation; allow a small positional/size tolerance.
            assert abs(x - expected_pos[0]) <= 2
            assert abs(y - expected_pos[1]) <= 2
            assert abs(w - expected_size[0]) <= 2
            assert abs(h - expected_size[1]) <= 2
            assert score >= 0.8

    def test_search_scales_configurable_per_call_and_per_instance(self):
        template = make_template()
        scaled = template.resize((60, 60), Image.LANCZOS)
        canvas = paste(make_canvas(), scaled, (40, 40))

        default_handler = ImageHandler(debug_mode=False, match_scales=(1.0,))
        assert default_handler.find_image_matches(template, canvas) == []

        configured = ImageHandler(
            debug_mode=False, match_scales=(1.0, 1.5), match_threshold=0.85
        )
        assert len(configured.find_image_matches(template, canvas)) == 1
        # Per-call overrides must take precedence over instance configuration.
        assert len(default_handler.find_image_matches(template, canvas, scales=(1.5,), threshold=0.85)) == 1


class TestNonMaximumSuppression:
    def test_overlapping_candidates_collapse_to_single_hit(self, handler):
        template = make_template()
        pos = (70, 60)
        canvas = paste(make_canvas(), template, pos)

        # A dense scale set produces many overlapping raw peaks around one
        # real control; NMS must leave exactly one unique hit.
        matches = handler.find_image_matches(
            template, canvas,
            scales=(0.9, 0.95, 1.0, 1.05, 1.1),
            threshold=0.8,
            nms_threshold=0.3,
        )
        assert len(matches) == 1
        x, y, w, h, score = matches[0]
        assert abs(x - pos[0]) <= 2
        assert abs(y - pos[1]) <= 2

        # And one unique center point with a score.
        centers = handler.find_image_match_centers(
            template, canvas, scales=(0.9, 0.95, 1.0, 1.05, 1.1), threshold=0.8
        )
        assert len(centers) == 1
        cx, cy, center_score = centers[0]
        assert abs(cx - (pos[0] + template.size[0] // 2)) <= 3
        assert abs(cy - (pos[1] + template.size[1] // 2)) <= 3
        assert center_score == pytest.approx(score, abs=1e-9)

    def test_distinct_nearby_controls_are_not_merged(self, handler):
        template = make_template()
        pos_a, pos_b = (20, 30), (200, 150)
        canvas = paste(paste(make_canvas(), template, pos_a), template, pos_b)

        matches = handler.find_image_matches(template, canvas, scales=(1.0,))
        assert len(matches) == 2
        positions = {(m[0], m[1]) for m in matches}
        assert positions == {pos_a, pos_b}
        # No pair of surviving boxes may overlap beyond the NMS threshold.
        boxes = [m[:4] for m in matches]
        assert handler._iou(boxes[0], boxes[1]) == pytest.approx(0.0, abs=1e-9)


class TestThresholdBoundary:
    def test_match_at_exact_threshold_is_kept(self, handler):
        template = make_template()
        pos = (50, 50)
        canvas = paste(make_canvas(), template, pos)

        matches = handler.find_image_matches(template, canvas, scales=(1.0,), threshold=0.99)
        assert len(matches) == 1
        kept_score = matches[0][4]
        # The retained score must satisfy the >= threshold boundary rule.
        assert kept_score >= 0.99

    def test_threshold_above_best_score_returns_empty(self, handler):
        from PIL import ImageFilter

        template = make_template()
        # Blurring the pasted copy lowers the achievable correlation.
        degraded = template.filter(ImageFilter.GaussianBlur(radius=1.2))
        canvas = paste(make_canvas(), degraded, (50, 50))

        matches = handler.find_image_matches(template, canvas, scales=(1.0,), threshold=0.999)
        assert matches == []

    def test_invalid_threshold_falls_back_to_default(self, handler):
        template = make_template()
        canvas = paste(make_canvas(), template, (10, 10))
        # Invalid thresholds must be ignored rather than raising or matching all.
        assert len(handler.find_image_matches(template, canvas, scales=(1.0,), threshold=1.5)) == 1
        assert len(handler.find_image_matches(template, canvas, scales=(1.0,), threshold=-1)) == 1
        assert handler.find_image_matches(template, canvas, scales=(1.0,), threshold="nonsense") != []


class TestDualAxisScaling:
    def test_independent_x_and_y_scale_factors(self, handler):
        template = make_template(size=(40, 40))
        # 150% horizontally, 200% vertically: no isotropic scale fits.
        scaled = template.resize((60, 80), Image.LANCZOS)
        pos = (45, 35)
        canvas = paste(make_canvas(size=(300, 260)), scaled, pos)

        # Isotropic-only search must miss the anisotropically scaled control.
        assert handler.find_image_matches(
            template, canvas, scales=(1.0, 1.5, 2.0), threshold=0.85
        ) == []

        matches = handler.find_image_matches(
            template, canvas, scales=((1.5, 2.0),), threshold=0.85
        )
        assert len(matches) == 1
        x, y, w, h, score = matches[0]
        assert abs(x - pos[0]) <= 2
        assert abs(y - pos[1]) <= 2
        assert abs(w - 60) <= 2
        assert abs(h - 80) <= 2
        assert score >= 0.85


class TestRepeatedIcons:
    def test_same_icon_repeated_returns_one_unique_hit_each(self, handler):
        template = make_template(size=(32, 32))
        positions = [(15, 15), (120, 15), (15, 120), (120, 120), (230, 90)]
        canvas = make_canvas(size=(320, 200))
        for pos in positions:
            canvas = paste(canvas, template, pos)

        matches = handler.find_image_matches(template, canvas, scales=(1.0,))
        assert len(matches) == len(positions)
        assert {(m[0], m[1]) for m in matches} == set(positions)

        centers = handler.find_image_match_centers(template, canvas, scales=(1.0,))
        assert len(centers) == len(positions)
        expected_centers = {center_of((x, y) + template.size) for x, y in positions}
        assert {(c[0], c[1]) for c in centers} == expected_centers
        # All scores are finite and within the normalized correlation range.
        assert all(np.isfinite(c[2]) and 0.0 <= c[2] <= 1.0 for c in centers)


class TestConsecutiveCalls:
    def test_repeated_calls_are_deterministic_and_stateless(self, handler):
        template = make_template()
        canvas = make_canvas()
        for pos in ((30, 40), (180, 110)):
            canvas = paste(canvas, template, pos)

        first = handler.find_image_matches(
            template, canvas, scales=(0.9, 1.0, 1.1), threshold=0.85
        )
        for _ in range(4):
            again = handler.find_image_matches(
                template, canvas, scales=(0.9, 1.0, 1.1), threshold=0.85
            )
            assert again == first
        assert len(first) == 2

    def test_handler_instance_reusable_across_different_images(self, handler):
        t1, t2 = make_template(seed=1), make_template(seed=2)
        c1 = paste(make_canvas(seed=11), t1, (25, 25))
        c2 = paste(make_canvas(seed=12), t2, (90, 80))
        assert len(handler.find_image_matches(t1, c1, scales=(1.0,))) == 1
        assert len(handler.find_image_matches(t2, c2, scales=(1.0,))) == 1
        # Cross-matching different template/canvas pairs must not bleed state.
        assert handler.find_image_matches(t1, c2, scales=(1.0,), threshold=0.95) == []


class TestEdgeCases:
    def test_none_inputs_return_empty_or_none(self, handler):
        template = make_template()
        canvas = make_canvas()
        assert handler.find_image_matches(template, None) == []
        assert handler.find_image_matches(None, canvas) == []
        assert handler.find_image_matches(None, None) == []
        assert handler.find_image_match_centers(template, None) == []
        # Legacy single-hit API keeps returning None on invalid input.
        assert handler.find_image_location(template, None) is None
        assert handler.find_image_location(None, canvas) is None
        # Legacy batch API returns an empty list instead of None.
        assert handler.find_all_image_locations(template, None) == []
        assert handler.find_all_image_locations(None, canvas) == []

    def test_empty_images_return_empty_without_raising(self, handler):
        empty = Image.new("RGB", (0, 0))
        template = make_template()
        canvas = make_canvas()
        assert handler.find_image_matches(template, empty) == []
        assert handler.find_image_matches(empty, canvas) == []
        assert handler.find_image_matches(empty, empty) == []
        assert handler.find_all_image_locations(empty, canvas) == []
        assert handler.find_image_location(empty, canvas) is None

    def test_template_larger_than_parent_returns_empty(self, handler):
        template = make_template(size=(120, 120))
        canvas = make_canvas(size=(60, 60))
        assert handler.find_image_matches(template, canvas) == []
        assert handler.find_image_matches(template, canvas, scales=(0.5, 1.0, 2.0)) == []
        assert handler.find_all_image_locations(template, canvas) == []
        assert handler.find_image_location(template, canvas) is None

    def test_degenerate_constant_images_do_not_raise(self, handler):
        constant_template = Image.new("RGB", (20, 20), color=(128, 128, 128))
        # A constant template has zero variance: normalized correlation is
        # undefined, so it must never be matched, regardless of the canvas.
        noisy_canvas = make_canvas(seed=31)
        assert handler.find_image_matches(constant_template, noisy_canvas) == []
        assert handler.find_image_match_centers(constant_template, noisy_canvas) == []

        constant_canvas = Image.new("RGB", (80, 80), color=(128, 128, 128))
        assert handler.find_image_matches(constant_template, constant_canvas, scales=(1.0,)) == []
        assert handler.find_all_image_locations(constant_template, constant_canvas) == []
        assert handler.find_image_location(constant_template, constant_canvas) is None

    def test_numpy_array_inputs_are_accepted(self, handler):
        template = make_template()
        canvas = paste(make_canvas(), template, (33, 44))
        matches = handler.find_image_matches(np.array(template), np.array(canvas), scales=(1.0,))
        assert len(matches) == 1
        assert (matches[0][0], matches[0][1]) == (33, 44)


class TestLegacyApiCompatibility:
    def test_find_image_location_returns_single_four_tuple(self, handler):
        template = make_template()
        pos = (55, 65)
        canvas = paste(make_canvas(), template, pos)

        location = handler.find_image_location(template, canvas)
        assert location is not None
        assert len(location) == 4
        x, y, w, h = location
        assert (x, y) == pos
        assert (w, h) == template.size

    def test_find_all_image_locations_deduplicated_four_tuples(self, handler):
        template = make_template(size=(32, 32))
        positions = [(20, 20), (150, 100)]
        canvas = make_canvas()
        for pos in positions:
            canvas = paste(canvas, template, pos)

        locations = handler.find_all_image_locations(template, canvas)
        assert isinstance(locations, list)
        assert len(locations) == 2
        assert all(len(loc) == 4 for loc in locations)
        assert {(loc[0], loc[1]) for loc in locations} == set(positions)

    def test_no_match_legacy_apis(self, handler):
        template = make_template(seed=99)
        canvas = make_canvas(seed=100)
        assert handler.find_image_location(template, canvas) is None
        assert handler.find_all_image_locations(template, canvas) == []


class TestRPALiteCoordinateConversion:
    """RPALite-level tests run with the headless stubs from conftest.py and
    bypass display-scaling detection so the conversion math is deterministic."""

    def _make_rpalite(self, scale_x, scale_y):
        from RPALite.rpalite import RPALite

        rpa = RPALite.__new__(RPALite)
        rpa.platform = "Linux"
        rpa.debug_mode = False
        rpa.image_handler = ImageHandler(debug_mode=False)
        rpa._display_scale_factor_x = scale_x
        rpa._display_scale_factor_y = scale_y
        return rpa

    def test_physical_matches_convert_back_to_logical_screen(self):
        rpa = self._make_rpalite(1.5, 1.5)
        template = make_template(size=(40, 40))
        # Physical screenshot contains the icon at 150% scaling.
        scaled = template.resize((60, 60), Image.LANCZOS)
        physical_pos = (150, 90)
        canvas = paste(make_canvas(size=(320, 240)), scaled, physical_pos)

        physical = rpa.find_image_matches(template, canvas, scales=(1.5,), threshold=0.85)
        assert len(physical) == 1
        px, py, pw, ph, score = physical[0]
        assert abs(px - physical_pos[0]) <= 2

        logical = rpa.find_image_matches(
            template, canvas, scales=(1.5,), threshold=0.85,
            return_logical_coordinates=True,
        )
        assert len(logical) == 1
        lx, ly, lw, lh, logical_score = logical[0]
        assert logical_score == pytest.approx(score, abs=1e-9)
        assert lx == pytest.approx(px / 1.5, abs=1)
        assert ly == pytest.approx(py / 1.5, abs=1)
        assert lw == pytest.approx(pw / 1.5, abs=1)
        assert lh == pytest.approx(ph / 1.5, abs=1)

    def test_logical_centers_under_dual_axis_scaling(self):
        rpa = self._make_rpalite(1.5, 2.0)
        template = make_template(size=(40, 40))
        scaled = template.resize((60, 80), Image.LANCZOS)
        physical_pos = (120, 160)
        canvas = paste(make_canvas(size=(320, 320)), scaled, physical_pos)

        centers = rpa.find_image_match_centers(
            template, canvas, scales=((1.5, 2.0),), threshold=0.85,
            return_logical_coordinates=True,
        )
        assert len(centers) == 1
        cx, cy, score = centers[0]
        physical_cx = physical_pos[0] + 60 // 2
        physical_cy = physical_pos[1] + 80 // 2
        assert cx == pytest.approx(physical_cx / 1.5, abs=1)
        assert cy == pytest.approx(physical_cy / 2.0, abs=1)
        assert score >= 0.85

    def test_rpalite_constructor_forwards_matching_config(self):
        from RPALite.rpalite import RPALite

        rpa = RPALite.__new__(RPALite)
        rpa.platform = "Linux"
        rpa.debug_mode = False
        rpa.image_handler = ImageHandler(
            debug_mode=False,
            match_threshold=0.82,
            match_scales=(1.0, (1.5, 2.0)),
            nms_threshold=0.4,
        )
        rpa._display_scale_factor_x = 1.0
        rpa._display_scale_factor_y = 1.0
        assert rpa.image_handler.match_threshold == 0.82
        assert rpa.image_handler.nms_threshold == 0.4
        assert rpa.image_handler.match_scales == ((1.0, 1.0), (1.5, 2.0))
