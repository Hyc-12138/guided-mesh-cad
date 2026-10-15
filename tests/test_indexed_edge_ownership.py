import unittest
import cadquery as cq
from guided_mesh_cad.cad import _native_edge_face_owners,validate_shared_boundaries,PatchTopologyError


class IndexedEdgeOwnershipTests(unittest.TestCase):
    def assert_matches_native_identity(self,faces,probes):
        identities,owners=_native_edge_face_owners(faces)
        for probe in probes:
            expected=[i for i,f in enumerate(faces,1) if any(probe.wrapped.IsSame(e.wrapped) for e in f.Edges())]
            self.assertEqual(owners.get(identities.FindIndex(probe.wrapped),[]),expected)

    def test_box_cylinder_seam_and_reversed_edges_match(self):
        for solid in [cq.Solid.makeBox(2,3,4),cq.Solid.makeCylinder(3,7)]:
            probes=solid.Edges()+[e.reverse() for e in solid.Edges()]
            self.assert_matches_native_identity(solid.Faces(),probes)

    def test_locations_and_coincident_independent_edges_are_distinct(self):
        first=cq.Solid.makeBox(2,3,4);located=first.moved(cq.Location(cq.Vector(10,0,0)))
        independent=cq.Solid.makeBox(2,3,4)
        faces=first.Faces()+located.Faces()+independent.Faces()
        self.assert_matches_native_identity(faces,first.Edges()+located.Edges()+independent.Edges())
        ids,_=_native_edge_face_owners(faces)
        self.assertNotEqual(ids.FindIndex(first.Edges()[0].wrapped),ids.FindIndex(located.Edges()[0].wrapped))
        self.assertNotEqual(ids.FindIndex(first.Edges()[0].wrapped),ids.FindIndex(independent.Edges()[0].wrapped))

    def test_duplicate_input_faces_preserve_positional_owner_semantics(self):
        face=cq.Solid.makeBox(2,3,4).Faces()[0]
        self.assert_matches_native_identity([face,face],face.Edges())

    def test_duplicate_declared_edge_retains_original_error_owner(self):
        edge=cq.Edge.makeLine((0,0,0),(3,0,0))
        declaration={'edge_a':edge,'edge_b':edge.reverse(),'alignment':'reverse','samples':3}
        with self.assertRaisesRegex(PatchTopologyError,'already belongs to first') as result:
            validate_shared_boundaries([dict(declaration,id='first'),dict(declaration,id='second')],1e-7)
        self.assertEqual(len(result.exception.report['shared_boundaries']),2)


if __name__=='__main__':unittest.main()
