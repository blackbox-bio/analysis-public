# change the column names to be more descriptive:

# change luminance to pressure-index
# change luminescence to luminance
# change paws to LHP, RHP, LFP, RFP
# change shoulder to elbow
# change l/r-hip to l/r-femur
# change hip to sacrum
# the hip now refers to tailbase-sacrum line
# change ankle to heel
# change the sacrum-tailbase-(L/R)HP angle to (L/R)HP_limb-angle
# change sternumhead to sternal-notch
# change sternumtail to xyphoid
# (see ./name_dicts.json for all changes)

import json

_f = open('name_dicts.json')
_name_dicts = json.load(_f)

# ---------- summary_col_name_dict ----------
summary_col_name_dict: dict[str, str] = _name_dicts["summary_col_name"]


# ---------- features_col_name_dict ----------
features_col_name_dict: dict[str, str] = _name_dicts["features_col_name"]

_f.close()
