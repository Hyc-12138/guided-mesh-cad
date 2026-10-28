"""Native trim classification policy shared by all actual-boundary paths.

OCCT 7.9.3 theGapCheckTol is the vertex-tolerance threshold that triggers
special gap-intersection handling, not the point classification precision.
Use its documented default. Passing a tiny value activates that special
handling on ordinary native vertices and can turn true material into OUT.
The query tolerance, bbox strategy and callers' IN/ON/OUT handling stay fixed.
No shape, topology tolerance, fuzzy value or physical acceptance limit changes.
"""
from OCP.gp import gp_Pnt2d

def classify_native_uv(classifier, face, u, v):
    classifier.Perform(face, gp_Pnt2d(u, v), 1e-9, True, 0.1)
    return classifier.State()
