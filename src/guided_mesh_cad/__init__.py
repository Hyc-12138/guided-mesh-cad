"""Guided reconstruction; no automatic feature recognition or CAD-history recovery."""
__version__ = "0.3.0.dev23+public.1"

def __getattr__(name):
    # CLI status, resume planning and stored diagnostics need no numeric/CAD runtime.
    if name in ('resample_closed_contours_by_anchors', 'simplify_corresponding_polylines'):
        from importlib import import_module
        value = getattr(import_module('.charts', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)

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


def union_positive_solids(*args, **kwargs):
    """Explicit serial native union with per-step positive-material checks."""
    from .positive_union import union_positive_solids as union
    return union(*args, **kwargs)


def validate_positive_union(*args, **kwargs):
    """Check named original positive inputs are retained in a native result."""
    from .positive_union import validate_positive_union as validate
    return validate(*args, **kwargs)


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
           'extract_material_boundary', 'union_positive_solids', 'validate_positive_union',
           'prepare_source_disk_quads', 'make_control_face',
           'pullback_affine_pcurve', 'make_affine_pcurve_edge']



def build_winding_index(vertices, faces, *, leaf_size=24):
    """Index an unchanged triangle soup for bounded signed-winding queries.

    This classifies abs(w)>0.5, not physical occupancy of an open mesh.
    NumPy is sufficient; the optional acceleration extra enables Numba.
    """
    from .winding import BoundedWinding
    return BoundedWinding(vertices, faces, leaf_size=leaf_size)
