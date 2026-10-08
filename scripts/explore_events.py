import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import finlab
from finlab import data
import finlab.data as fd
import pandas as pd

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
fd._default_context._role = 'vip'

# 1. trading_halt (停損/停利)
print("=" * 60)
print("1. trading_halt (停損/停利/交易暫停)")
print("=" * 60)
halt_stock = data.get('trading_halt:暫停交易')
print(f"Type: {type(halt_stock)}")
print(f"Shape: {halt_stock.shape}")
print(f"Index sample: {halt_stock.index[:3]}")
if hasattr(halt_stock, 'columns'):
    print(f"Columns: {list(halt_stock.columns)[:5]}")
    print(halt_stock.iloc[:3, :5])
else:
    print(halt_stock.head(10))

print()
print("--- 暫停時間 ---")
halt_time = data.get('trading_halt:暫停交易時間')
print(f"Shape: {halt_time.shape}")
print(halt_time.head(5))

print()
print("--- 市場別 ---")
market = data.get('trading_halt:市場別')
print(f"Unique values: {market.stack().unique()[:10]}")

print()
print("--- 有價證券類別 ---")
category = data.get('trading_halt:有價證券類別')
print(f"Unique values: {category.stack().unique()[:10]}")

# 2. day_trade_short_suspension (當沖放空暫停)
print()
print("=" * 60)
print("2. day_trade_short_suspension (當沖放空暫停)")
print("=" * 60)
try:
    dts = data.get('day_trade_short_suspension')
    print(f"Type: {type(dts)}")
    print(f"Shape: {dts.shape}")
    if hasattr(dts, 'columns'):
        print(f"Columns: {list(dts.columns)}")
        print(dts.head(5))
    else:
        print(dts.head(10))
except Exception as e:
    print(f"Error: {e}")

# 3. margin_short_sale_suspension (融券放空暫停)
print()
print("=" * 60)
print("3. margin_short_sale_suspension (融券放空暫停)")
print("=" * 60)
try:
    mss = data.get('margin_short_sale_suspension')
    print(f"Type: {type(mss)}")
    print(f"Shape: {mss.shape}")
    if hasattr(mss, 'columns'):
        print(f"Columns: {list(mss.columns)}")
        print(mss.head(5))
    else:
        print(mss.head(10))
except Exception as e:
    print(f"Error: {e}")

# 4. important_info_announcement (重大訊息)
print()
print("=" * 60)
print("4. important_info_announcement (重大訊息)")
print("=" * 60)
try:
    ia = data.get('important_info_announcement')
    print(f"Type: {type(ia)}")
    print(f"Shape: {ia.shape}")
    if hasattr(ia, 'columns'):
        print(f"Columns: {list(ia.columns)}")
        print(ia.head(5))
    else:
        print(ia.head(10))
except Exception as e:
    print(f"Error: {e}")

# 5. Check esb_attention_disposal (VIP enhanced)
print()
print("=" * 60)
print("5. esb_attention_disposal (VIP enhanced)")
print("=" * 60)
try:
    esb = data.get('esb_attention_disposal:處置條件')
    print(f"Shape: {esb.shape}")
    print(esb.head(5))
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("SUMMARY: Event types available")
print("=" * 60)
print("""
1. trading_halt - 停損/停利 (price limit halt)
2. day_trade_short_suspension - 當沖放空暫停
3. margin_short_sale_suspension - 融券放空暫停  
4. important_info_announcement - 重大訊息公告
5. esb_attention_disposal - 注意/處置 (VIP enhanced)
""")
