# Data

Nothing is committed here. Expected layout for real authenticity data:

```
data/authenticity/
  train/{genuine,counterfeit}/*.jpg
  val/{genuine,counterfeit}/*.jpg
  test/{genuine,counterfeit}/*.jpg
```

Rules that matter more than model choice:
1. **Split by item, not by photo.** Several photos of one bag must all land in the same split.
2. **Match the conditions.** If counterfeits are all from marketplace listings and genuines from brand
   websites, the model learns the photo style, not authenticity. Grad-CAM will expose this.
3. **Only use data you're licensed to use**, and document its provenance and how labels were verified.
4. Keep a test set untouched until the very end.

For a smoke test without real data: `python scripts/make_demo_dataset.py` (synthetic, meaningless images).
