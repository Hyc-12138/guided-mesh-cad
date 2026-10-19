"""Guided reconstruction; no automatic feature recognition or CAD-history recovery."""
__version__ = "0.3.0.dev17+public.1"

from .charts import resample_closed_contours_by_anchors, simplify_corresponding_polylines

def validate_native_boundary(*args, **kwargs):
    """Lazy entry point for the explicit actual trimmed-boundary metric."""
    from .boundary_validation import validate_native_boundary as validate
    return validate(*args, **kwargs)

def sample_native_edge_bounded(*args, **kwargs):
    """Lazy entry point for derivative-controlled actual native curve sampling."""
    from .curve_sampling import sample_native_edge_bounded as sample
    return sample(*args, **kwargs)


def validate_native_boundary_low_memory(*args, **kwargs):
    """Complete actual-boundary metric using explicit low-peak disk storage."""
    from .boundary_validation import validate_native_boundary_low_memory as validate
    return validate(*args, **kwargs)

def validate_native_boundary_uv_witness(*args, **kwargs):
    """Explicit actual native point-witness strategy, with original fallback."""
    from .boundary_validation import validate_native_boundary_uv_witness as validate
    return validate(*args, **kwargs)

def extract_material_boundary(*args, **kwargs):
    """Explicit, contained-INTERNAL-sheet extraction; default gates stay strict."""
    from .material_boundary import extract_material_boundary as extract
    return extract(*args, **kwargs)


def prepare_source_disk_quads(*args, **kwargs):
    """Prepare complete source-disk quad charts with original barycentric lineage."""
    from .disk_charts import prepare_source_disk_quads as prepare
    return prepare(*args, **kwargs)


def make_control_face(*args, **kwargs):
    """Explicit C2 control-net approximation, distinct from exact interpolation."""
    from .freeform_control import make_control_face as make
    return make(*args, **kwargs)

def pullback_affine_pcurve(*args, **kwargs):
    """Certified floating whole-span rational pullback of one affine UV path."""
    from .affine_pcurve import pullback_affine_pcurve as pullback
    return pullback(*args, **kwargs)


def make_affine_pcurve_edge(*args, **kwargs):
    """Native 3D edge with its same-parameter affine support pcurve."""
    from .affine_pcurve import make_affine_pcurve_edge as make
    return make(*args, **kwargs)


__all__ = ['resample_closed_contours_by_anchors', 'simplify_corresponding_polylines',
           'validate_native_boundary', 'sample_native_edge_bounded',
           'validate_native_boundary_uv_witness', 'validate_native_boundary_low_memory',
           'extract_material_boundary', 'prepare_source_disk_quads', 'make_control_face',
           'pullback_affine_pcurve', 'make_affine_pcurve_edge']

