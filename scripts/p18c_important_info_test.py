"""P18c: Quick check on important_info_announcement (重大訊息) feasibility."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

print("=== important_info_announcement ===")
try:
    ia = data.get('important_info_announcement')
    print(f"Type: {type(ia)}")
    print(f"Shape: {ia.shape}")
    if hasattr(ia, 'columns'):
        print(f"Columns: {list(ia.columns)}")
        print(ia.head(5).to_string())
        print(f"\nDate range: {ia.index.min()} to {ia.index.max()}")
        if 'reason' in ia.columns or '原因' in ia.columns:
            col = 'reason' if 'reason' in ia.columns else '原因'
            print(f"\nTop reasons:")
            print(ia[col].value_counts().head(20).to_string())
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()
print("=== day_trade_short_suspension reasons ===")
dts = data.get('day_trade_short_suspension')
print(f"Reasons:")
print(dts['原因'].value_counts().head(10).to_string())
