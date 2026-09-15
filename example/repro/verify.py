"""Verify the scoped export without running experiments."""
import json
from capsule import check
if __name__=='__main__':
 result=check();print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
