"""CPU witnesses for tail coverage, past-only initialization and equal DDP batches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from codetrack.frame_coverage import SequenceMetadata,build_plan,indices_for,shard_plan


def sequence(count,invalid=()):
    boxes=np.tile([10.,20.,30.,40.],(count,1))
    for index in invalid:boxes[index,2:]=0
    return SequenceMetadata('video',boxes,tuple(Path(str(i)) for i in range(count)),
                            tuple(Path(str(i)) for i in range(count)),'annotation-hash')


class CoverageTests(unittest.TestCase):
    def test_all_lengths_and_remainders(self):
        for length in (1,8,16):
            for n in range(length+1,length*5+8):
                metadata=sequence(n,invalid=range(1,min(n-1,9)))
                plan,report=build_plan([metadata],length)
                visits=np.zeros(n,dtype=int)
                for row in plan:
                    _,start,initial,span=row
                    self.assertLess(initial,start)
                    self.assertTrue(metadata.valid[initial])
                    visits[start:start+span]+=1
                    indices,mask=indices_for(row,n)
                    self.assertEqual(indices[0],initial)
                    self.assertEqual(indices[1:1+span],list(range(start,start+span)))
                    self.assertTrue(mask[:span].all())
                    self.assertTrue(all(0<=i<n for i in indices))
                    self.assertEqual(len(indices),1+length+(3 if length>1 else 0))
                self.assertEqual(visits[0],0)
                self.assertTrue((visits[1:]>=1).all())
                self.assertEqual(report['unique_search_frames_covered'],n-1)

    def test_long_invalid_run_does_not_erase_search_frames(self):
        plan,report=build_plan([sequence(83,invalid=range(5,62))],8)
        self.assertEqual(report['invalid_search_annotations'],57)
        self.assertGreater(report['per_sequence'][0]['max_initialization_gap_frames'],8)
        self.assertTrue(any(int(row[2])==4 and int(row[1])>12 for row in plan))

    def test_shards_equal_batches_complete_union_padding_only(self):
        for count in (2,7,19,100,151):
            plan=np.zeros((count,4),dtype=np.int64)
            for batch in (1,4,16):
                shards=[shard_plan(plan,3,rank,2,batch) for rank in range(2)]
                a,b=(s[0] for s in shards)
                self.assertEqual(len(a),len(b));self.assertEqual(len(a)%batch,0)
                all_ids=np.concatenate((a,b))
                self.assertTrue(np.array_equal(np.unique(all_ids),np.arange(count)))
                self.assertEqual(len(all_ids)-count,shards[0][1])
                self.assertTrue(np.array_equal(a,shard_plan(plan,3,0,2,batch)[0]))

    def test_tail_future_mask(self):
        plan,_=build_plan([sequence(28)],8)
        _,available=indices_for(plan[-1],28)
        self.assertEqual(available.tolist(),[True]*8+[False]*3)
        windows=np.stack([available[t:t+4] for t in range(8)])
        self.assertEqual(windows.all(1).tolist(),[True]*5+[False]*3)

    def test_annotation_changes_change_fingerprint(self):
        a=sequence(40);b=sequence(40);b.annotation_sha256='changed'
        self.assertNotEqual(build_plan([a],8)[1]['plan_sha256'],build_plan([b],8)[1]['plan_sha256'])


if __name__=='__main__':unittest.main()
