from __future__ import annotations
import hashlib, json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
class SplitTests(unittest.TestCase):
    def test_darcy(self):
        p=ROOT/"splits/darcy_beta1.0__seed0001__N8000_1000_1000.json"; x=json.loads(p.read_text(encoding="utf-8-sig")); self.assertEqual((len(x["train"]),len(x["val"]),len(x["test"])),(8000,1000,1000)); self.assertEqual(len(set(x["train"])|set(x["val"])|set(x["test"])),10000); self.assertFalse(set(x["train"])&set(x["val"])); self.assertFalse(set(x["train"])&set(x["test"])); self.assertFalse(set(x["val"])&set(x["test"])); self.assertEqual(digest(p),"FBCD5D6DE95BB0BD80ADCBAB0BDAB245FA7CA5E97B8CDC1FDD0853F538DC5B28")
    def test_hyperelasticity(self):
        p=ROOT/"splits/vino_hyperelasticity_split_seed0001_N400_75_75.json"; x=json.loads(p.read_text(encoding="utf-8-sig"))["canonical_payload"]; a,b,c=map(set,(x["train_indices"],x["validation_indices"],x["test_indices"])); self.assertEqual((len(a),len(b),len(c)),(400,75,75)); self.assertEqual(len(a|b|c),550); self.assertFalse(a&b); self.assertFalse(a&c); self.assertFalse(b&c); self.assertTrue(all(0<=i<550 for i in a|b|c)); self.assertEqual(digest(p),"35372BD892CAA8D93150812EB001E9A214DAF91BDE8893E43EC4EB17DA34A8F6")
if __name__=="__main__": unittest.main()
