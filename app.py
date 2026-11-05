# ==============================================================
# SHOPFLOW v2 - E-COMMERCE ORDER PROCESSING SIMULATION
# NEC Course: Solving Problems Using Modelling and Simulation (NEC00091)
#
# Run:   pip install -r requirements.txt
#        streamlit run app.py
#
# Model idea:  Entities = customers, orders, products
#              State    = order status, stock, payment status
#              Events   = place order, pay, pack, ship, deliver, cancel
#              Randomness = arrivals (Poisson), payment failure, cancellations
# ==============================================================

import copy
import html
import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ShopFlow | Order Control Center", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

/* ---------- Base: main area always light ---------- */
.stApp { background:#f3f5fa; color:#101828; color-scheme:light; }
[data-testid="stHeader"] { background:#f3f5fa; }
[data-testid="stMain"] label, [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stMain"] h2, [data-testid="stMain"] h3 { color:#101828 !important; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] { color:#667085 !important; }

/* ---------- Inputs, selects, number fields ---------- */
[data-testid="stMain"] [data-baseweb="input"], [data-testid="stMain"] [data-baseweb="base-input"],
[data-testid="stMain"] [data-baseweb="select"] > div { background:#fff !important; border-color:#d0d5dd !important; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea { color:#101828 !important; -webkit-text-fill-color:#101828 !important; background:transparent !important; }
[data-testid="stMain"] [data-baseweb="select"] div { color:#101828 !important; }
[data-testid="stMain"] [data-testid="stNumberInput"] button { background:#f2f4f7 !important; color:#101828 !important; }
[data-testid="stMain"] [data-testid="stSlider"] div { color:#101828 !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul { background:#fff !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color:#101828 !important; }

/* ---------- Buttons ---------- */
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] { background:#fff !important; border:1px solid #d0d5dd !important; }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] p { color:#101828 !important; }
[data-testid="stMain"] [data-testid="stBaseButton-primary"],
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button { background:#2563eb !important; border:none !important; }
[data-testid="stMain"] [data-testid="stBaseButton-primary"] p { color:#fff !important; }
div.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:700; }

/* ---------- Tabs, expander, containers ---------- */
[data-testid="stMain"] [data-testid="stTabs"] button p { color:#344054 !important; font-weight:700; }
[data-testid="stMain"] [data-testid="stExpander"] { background:#fff; border-radius:12px; }
[data-testid="stMain"] [data-testid="stExpander"] summary p, [data-testid="stMain"] [data-testid="stExpander"] summary span { color:#101828 !important; }
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:14px; }

/* ---------- Sidebar: always dark ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1220,#1b2545); }
[data-testid="stSidebar"] * { color:#e8ecff !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] { background:rgba(255,255,255,.10) !important; border:1px solid rgba(255,255,255,.30) !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p { color:#fff !important; }

/* ---------- Custom components ---------- */
.hero { background:linear-gradient(120deg,#1d4ed8,#6d28d9); border-radius:20px; padding:22px 28px; margin-bottom:16px; }
[data-testid="stMain"] .hero h1 { color:#fff !important; font-size:29px; font-weight:800; margin:0; padding:0; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] .hero p { color:#dbe4ff !important; margin:4px 0 0; font-size:14px; }
.card { background:#fff; color:#101828; border:1px solid #e4e9f2; border-radius:16px; padding:16px 18px; }
.k { background:#fff; border:1px solid #e4e9f2; border-left:5px solid var(--c,#2563eb); border-radius:14px; padding:14px 16px; min-height:104px; }
.k-l { color:#667085; font-size:12px; font-weight:700; }
.k-v { color:#101828; font-size:25px; font-weight:800; margin-top:5px; }
.k-n { color:#667085; font-size:11.5px; margin-top:2px; }
.sec { color:#101828; font-size:18px; font-weight:800; margin:14px 0 8px; }
.note { background:#eef4ff; border:1px solid #d6e4ff; color:#1849a9; padding:12px 14px; border-radius:12px; font-size:14px; }
.stepper { display:flex; }
.step { flex:1; text-align:center; position:relative; }
.step .dot { width:44px; height:44px; border-radius:50%; margin:0 auto; display:flex; align-items:center; justify-content:center;
             font-size:19px; background:#eef1f6; border:2px solid #dde3ee; position:relative; z-index:2; }
.step .lbl { font-size:11.5px; font-weight:700; margin-top:7px; color:#667085; }
.step:not(:last-child)::after { content:""; position:absolute; top:21px; left:calc(50% + 26px); width:calc(100% - 52px); height:3px; background:#e0e5ee; }
.step.done .dot { background:#12b76a; border-color:#12b76a; color:#fff; font-weight:800; }
.step.done:not(:last-child)::after { background:#12b76a; }
.step.current .dot { background:#2563eb; border-color:#2563eb; box-shadow:0 0 0 6px rgba(37,99,235,.18); }
.step.current .lbl { color:#2563eb; }
.step.fail .dot { background:#f04438; border-color:#f04438; color:#fff; }
.step.fail .lbl { color:#f04438; }
.flow { background:#fff; border:1px solid #e4e9f2; border-radius:14px; padding:14px 16px; margin-bottom:10px; }
.flow b { color:#101828; font-size:15px; }
.flow span { color:#475467; font-size:13.5px; display:block; margin-top:3px; }
</style>""", unsafe_allow_html=True)


def H(s):
    """Render HTML safely (single line so markdown never treats it as code)."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


# --------------------------------------------------------------
# 1. CONSTANTS - products, stages, payment, cities, couriers
# --------------------------------------------------------------

PRODUCTS = [
    {"id": "P101", "name": "Laptop", "emoji": "💻", "category": "Electronics", "price": 50000, "stock": 12, "sold": 8},
    {"id": "P102", "name": "Smartphone", "emoji": "📱", "category": "Electronics", "price": 22000, "stock": 18, "sold": 15},
    {"id": "P103", "name": "Headphones", "emoji": "🎧", "category": "Accessories", "price": 1800, "stock": 30, "sold": 21},
    {"id": "P104", "name": "Mechanical Keyboard", "emoji": "⌨️", "category": "Accessories", "price": 2500, "stock": 22, "sold": 11},
    {"id": "P105", "name": "Wireless Mouse", "emoji": "🖱️", "category": "Accessories", "price": 900, "stock": 40, "sold": 26},
    {"id": "P106", "name": "Smart Watch", "emoji": "⌚", "category": "Wearables", "price": 3500, "stock": 14, "sold": 9},
]

# stage -> (icon, what customer sees, what the system does behind the scenes)
STAGE_META = {
    "Placed": ("🛒", "Order confirmation page",
               "Cart is validated, stock is checked and reserved, price (coupon + GST + shipping) is locked and an order ID is generated."),
    "Payment Verified": ("💳", "Payment success message (or COD confirmation)",
                         "Payment gateway contacts the bank / UPI app, runs a fraud check and gets authorisation. Invoice is generated and SMS + email are sent."),
    "Packed": ("📦", "'Your order is packed'",
               "Warehouse staff pick the item from the shelf, quality check is done, item is packed and the shipping label is printed."),
    "Shipped": ("🚚", "Courier name + tracking ID",
                "Parcel is handed over to a courier partner. Manifest is created and a tracking ID is issued."),
    "In Transit": ("🏭", "Live tracking updates",
                   "Parcel travels through sorting hubs and the destination city hub."),
    "Out for Delivery": ("🛵", "'Arriving today' + delivery OTP",
                         "Local delivery agent loads the parcel and the customer receives a delivery OTP."),
    "Delivered": ("✅", "Delivered + rate your product",
                  "OTP is verified, COD cash is collected (if any), order is closed and a review request is sent."),
}
FLOW = list(STAGE_META)
BAD = {"Payment Failed": "❌", "Cancelled": "🚫"}
ICON = {**{s: m[0] for s, m in STAGE_META.items()}, **BAD}

PAY_ONLINE = ["UPI", "Credit / Debit Card", "Net Banking", "Wallet"]
PAY_ALL = PAY_ONLINE + ["Cash on Delivery"]
CITIES = {"Jabalpur": 4, "Bhopal": 3, "Indore": 3, "Delhi": 2, "Mumbai": 2,
          "Bengaluru": 2, "Pune": 2, "Hyderabad": 3, "Chennai": 3, "Kolkata": 3}  # city -> delivery days
COURIERS = ["Delhivery", "Blue Dart", "Ecom Express", "XpressBees", "India Post"]
CUSTOMERS = ["Aarav Sharma", "Priya Verma", "Rohan Patel", "Ananya Gupta", "Kabir Singh", "Isha Jain"]
COUPONS = {"WELCOME10": 10, "SAVE5": 5, "FESTIVE15": 15}


# --------------------------------------------------------------
# 2. CORE MODEL - works on a "store" dict so it can run on a sandbox copy
# --------------------------------------------------------------

def quote(price, qty, coupon=""):
    """Price breakdown: subtotal - coupon + 18% GST + shipping (free above Rs 999)."""
    sub = price * qty
    pct = COUPONS.get(coupon.strip().upper(), 0)
    disc = round(sub * pct / 100)
    gst = round((sub - disc) * 0.18)
    ship = 0 if sub - disc >= 999 else 49
    return {"subtotal": sub, "discount": disc, "gst": gst, "shipping": ship,
            "total": sub - disc + gst + ship, "pct": pct}


def log(order, now, actor, text):
    order["events"].append({"time": now, "actor": actor, "event": text})


def get_product(store, pid):
    return next(p for p in store["products"] if p["id"] == pid)


def create_order(store, pid, qty, method, city, coupon, customer, now):
    p = get_product(store, pid)
    if qty < 1:
        return False, "Quantity must be at least 1.", None
    if p["stock"] < qty:
        return False, f"Insufficient stock. Only {p['stock']} unit(s) available.", None
    q = quote(p["price"], qty, coupon)
    p["stock"] -= qty
    p["sold"] += qty
    o = {"id": store["next_id"], "customer": customer, "city": city, "product_id": pid,
         "product": p["name"], "emoji": p["emoji"], "category": p["category"], "qty": qty,
         "subtotal": q["subtotal"], "discount": q["discount"], "gst": q["gst"],
         "shipping": q["shipping"], "total": q["total"], "coupon": coupon.strip().upper() if q["pct"] else "",
         "payment_method": method, "payment_status": "Pending", "status": "Placed", "reached": 0,
         "courier": "-", "tracking": "-", "eta": None, "created": now, "updated": now,
         "delivered_at": None, "events": []}
    store["next_id"] += 1
    store["orders"].insert(0, o)
    log(o, now, "Customer", f"Placed order for {qty} x {p['name']} from {city}. Total ₹{q['total']:,}.")
    log(o, now, "Inventory", f"{qty} unit(s) reserved. Stock left: {p['stock']}.")
    return True, f"Order #{o['id']} placed successfully.", o


def release_stock(store, o):
    p = get_product(store, o["product_id"])
    p["stock"] += o["qty"]
    p["sold"] -= o["qty"]


def advance(store, o, now, fail_rate=0.0):
    """Move one order to its next stage. Returns (ok, message)."""
    if o["status"] not in FLOW or o["status"] == "Delivered":
        return False, f"Order #{o['id']} cannot move forward ({o['status']})."
    nxt = FLOW[FLOW.index(o["status"]) + 1]
    cod = o["payment_method"] == "Cash on Delivery"
    if nxt == "Payment Verified":
        if cod:
            o["payment_status"] = "COD - pay on delivery"
            log(o, now, "System", "Cash on Delivery order verified. Payment will be collected at the doorstep.")
        else:
            log(o, now, "Payment Gateway", f"Payment request of ₹{o['total']:,} sent via {o['payment_method']}.")
            if random.random() < fail_rate:
                o.update(status="Payment Failed", payment_status="Failed", updated=now)
                release_stock(store, o)
                log(o, now, "Payment Gateway", "Payment declined (bank timeout / low balance). Stock released.")
                return False, f"Payment failed for order #{o['id']}. Use 'Retry payment'."
            o["payment_status"] = "Paid"
            log(o, now, "Payment Gateway", "Bank authorised the payment. Invoice generated, SMS + email sent.")
    elif nxt == "Packed":
        log(o, now, "Warehouse", "Item picked, quality check passed, packed and label printed.")
    elif nxt == "Shipped":
        o["courier"] = random.choice(COURIERS)
        o["tracking"] = "SF" + str(random.randint(10**9, 10**10 - 1))
        o["eta"] = now + timedelta(days=CITIES[o["city"]])
        log(o, now, "Logistics", f"Handed to {o['courier']}. Tracking ID {o['tracking']}.")
    elif nxt == "In Transit":
        log(o, now, "Courier", "Parcel reached the sorting hub and is moving to the destination city.")
    elif nxt == "Out for Delivery":
        log(o, now, "Courier", f"Delivery agent is on the way to {o['city']}. Delivery OTP sent.")
    elif nxt == "Delivered":
        o["delivered_at"] = now
        if cod:
            o["payment_status"] = "Paid (cash collected)"
        log(o, now, "Courier", "OTP verified and parcel delivered." + (" Cash collected." if cod else ""))
    o.update(status=nxt, reached=FLOW.index(nxt), updated=now)
    return True, f"Order #{o['id']} moved to {nxt}."


def cancel_order(store, o, now):
    if o["status"] not in ("Placed", "Payment Verified", "Packed"):
        return False, "Orders can only be cancelled before they are shipped."
    release_stock(store, o)
    if o["payment_status"] == "Paid":
        o["payment_status"] = "Refunded"
        log(o, now, "Payment Gateway", f"Refund of ₹{o['total']:,} started (5-7 working days).")
    o.update(status="Cancelled", updated=now)
    log(o, now, "Customer", "Order cancelled. Stock returned to inventory.")
    return True, f"Order #{o['id']} cancelled."


def retry_payment(store, o, now):
    p = get_product(store, o["product_id"])
    if p["stock"] < o["qty"]:
        return False, "Product is out of stock now."
    p["stock"] -= o["qty"]
    p["sold"] += o["qty"]
    o.update(status="Placed", payment_status="Pending", updated=now)
    log(o, now, "Customer", "Retrying payment. Stock reserved again.")
    return True, f"Order #{o['id']} is ready for payment again."


def seed_store():
    store = {"products": [dict(p) for p in PRODUCTS], "orders": [], "next_id": 1001}
    now = datetime.now()
    # (product index, qty, payment, stages to advance (-1 = payment fails), city, days ago)
    specs = [(0, 1, "Credit / Debit Card", 6, "Delhi", 6), (1, 1, "UPI", 5, "Mumbai", 4),
             (2, 2, "UPI", 3, "Jabalpur", 3), (3, 1, "Net Banking", 1, "Bhopal", 1),
             (4, 1, "Cash on Delivery", 6, "Indore", 5), (5, 1, "Wallet", 4, "Pune", 2),
             (1, 1, "UPI", -1, "Chennai", 1), (2, 1, "UPI", 0, "Kolkata", 0)]
    for pi, qty, method, steps, city, days in specs:
        t = now - timedelta(days=days)
        _, _, o = create_order(store, PRODUCTS[pi]["id"], qty, method, city, "", random.choice(CUSTOMERS), t)
        if steps == -1:
            advance(store, o, t + timedelta(hours=1), fail_rate=1.0)
        for k in range(max(steps, 0)):
            advance(store, o, t + timedelta(hours=7 * (k + 1)), 0.0)
    return store


# --------------------------------------------------------------
# 3. SIMULATION ENGINE - hour-by-hour discrete-time model
# --------------------------------------------------------------

PROGRESS = {"Placed": .70, "Payment Verified": .55, "Packed": .45,
            "Shipped": .40, "In Transit": .30, "Out for Delivery": .50}  # chance to move forward each hour


def simulate(store, hours, rate, fail, cod, cancel_pct, restock, threshold, start):
    first_id = store["next_id"]
    arrived = lost = restocks = 0
    series = []
    for h in range(1, hours + 1):
        now = start + timedelta(hours=h)
        weights = [p["sold"] + 5 for p in store["products"]]
        for _ in range(np.random.poisson(rate)):                    # random customer arrivals
            arrived += 1
            p = random.choices(store["products"], weights)[0]
            qty = random.choices([1, 2, 3], [70, 22, 8])[0]
            method = "Cash on Delivery" if random.random() < cod / 100 else random.choice(PAY_ONLINE)
            ok, _, _ = create_order(store, p["id"], qty, method, random.choice(list(CITIES)),
                                    "", random.choice(CUSTOMERS), now)
            lost += 0 if ok else 1                                   # stock-out = lost sale
        active = 0
        for o in store["orders"]:
            if o["status"] in FLOW and o["status"] != "Delivered":
                if o["status"] in ("Placed", "Payment Verified", "Packed") and random.random() < cancel_pct / 400:
                    cancel_order(store, o, now)
                    continue
                if random.random() < PROGRESS[o["status"]]:
                    advance(store, o, now, fail / 100)
                active += o["status"] in FLOW and o["status"] != "Delivered"
        if restock:
            for p in store["products"]:
                if p["stock"] <= threshold:
                    p["stock"] += 25
                    restocks += 1
        mine = [o for o in store["orders"] if o["id"] >= first_id]
        series.append({"Hour": h, "In pipeline": active,
                       "Delivered (cumulative)": sum(o["status"] == "Delivered" for o in mine),
                       "Units in stock": sum(p["stock"] for p in store["products"])})
    mine = [o for o in store["orders"] if o["id"] >= first_id]
    done = [(o["delivered_at"] - o["created"]).total_seconds() / 3600 for o in mine if o["delivered_at"]]
    return {"store": store, "series": pd.DataFrame(series).set_index("Hour"), "arrived": arrived, "lost": lost,
            "accepted": len(mine), "delivered": len(done),
            "failed": sum(o["status"] == "Payment Failed" for o in mine),
            "cancelled": sum(o["status"] == "Cancelled" for o in mine),
            "revenue": sum(o["total"] for o in mine if o["status"] not in BAD),
            "avg_hours": float(np.mean(done)) if done else None, "restocks": restocks,
            "stages": [sum(o["status"] == s for o in mine) for s in list(STAGE_META) + list(BAD)]}


# --------------------------------------------------------------
# 4. UI HELPERS
# --------------------------------------------------------------

def money(v):
    return f"₹{v:,.0f}"


def header(title, sub):
    H(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>')


def kpi(col, label, value, note="", color="#2563eb"):
    with col:
        H(f'<div class="k" style="--c:{color}"><div class="k-l">{label}</div>'
          f'<div class="k-v">{value}</div><div class="k-n">{note}</div></div>')


def stepper(o):
    bad, r = o["status"] in BAD, o["reached"]
    out = ""
    for i, s in enumerate(FLOW):
        cls, dot = "pending", STAGE_META[s][0]
        if o["status"] == "Delivered" or i < r or (bad and i <= r):
            cls, dot = "done", "✓"
        elif i == r and not bad:
            cls = "current"
        elif o["status"] == "Payment Failed" and i == r + 1:
            cls, dot = "fail", "✕"
        out += f'<div class="step {cls}"><div class="dot">{dot}</div><div class="lbl">{s}</div></div>'
    return f'<div class="card"><div class="stepper">{out}</div></div>'


def orders_df(lst):
    return pd.DataFrame([{
        "Order": f"#{o['id']}", "Customer": o["customer"], "City": o["city"],
        "Product": f"{o['emoji']} {o['product']}", "Qty": o["qty"], "Total": o["total"],
        "Payment": f"{o['payment_method']} ({o['payment_status']})",
        "Status": f"{ICON[o['status']]} {o['status']}",
        "Progress": 100 if o["status"] == "Delivered" else round(o["reached"] / 6 * 100),
        "Placed on": o["created"].strftime("%d %b, %I:%M %p")} for o in lst])


TABLE_CFG = {"Total": st.column_config.NumberColumn(format="₹%d"),
             "Progress": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")}


def next_time(o):
    """Simulated clock: each stage takes a few hours."""
    return o["updated"] + timedelta(hours=random.randint(2, 9))


# --------------------------------------------------------------
# 5. SESSION STATE + SIDEBAR
# --------------------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = seed_store()
store = st.session_state.store
products, orders = store["products"], store["orders"]

with st.sidebar:
    st.markdown("## 📦 ShopFlow")
    st.caption("Order Control Center")
    page = st.radio("Workspace", ["📊 Overview", "🛒 Place Order", "🚚 Order Tracker", "📦 Inventory",
                                  "🧪 Simulation Lab", "📘 How It Works"]).split(" ", 1)[1]
    st.markdown("---")
    fail_pct = st.slider("Payment failure chance (%)", 0, 40, 8,
                         help="Chance that the bank / gateway declines a prepaid payment.")
    st.markdown("---")
    st.markdown("🟢 **Admin / Inventory Manager**")
    st.caption("NEC00091 • Python Simulation")
    if st.button("↻ Reset demo data", use_container_width=True):
        del st.session_state["store"]
        st.session_state.pop("sim", None)
        st.rerun()

alive = [o for o in orders if o["status"] not in BAD]
cnt = {s: sum(o["status"] == s for o in orders) for s in list(STAGE_META) + list(BAD)}
low = [p for p in products if p["stock"] <= 5]


# --------------------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------------------

if page == "Overview":
    header("Dashboard Overview", "Live view of orders, payments, deliveries and stock across the store.")
    paid = [o for o in orders if o["payment_status"].startswith("Paid")]
    online = [o for o in orders if o["payment_method"] != "Cash on Delivery"]
    ok_rate = (sum(o["status"] != "Payment Failed" for o in online) / len(online) * 100) if online else 100
    c = st.columns(6)
    kpi(c[0], "Total orders", len(orders), "All orders received")
    kpi(c[1], "Revenue", money(sum(o["total"] for o in alive)), "Excl. failed / cancelled", "#12b76a")
    kpi(c[2], "Avg. order value", money(np.mean([o["total"] for o in alive])) if alive else "₹0", "Per valid order", "#7c3aed")
    kpi(c[3], "Delivered", cnt["Delivered"], "Completed orders", "#12b76a")
    kpi(c[4], "Payment success", f"{ok_rate:.0f}%", "Prepaid orders", "#f79009")
    kpi(c[5], "Low stock items", len(low), "5 units or fewer", "#f04438")

    st.markdown('<div class="sec">Order journey - where are my orders right now?</div>', unsafe_allow_html=True)
    cols = st.columns(7)
    for col, s in zip(cols, FLOW):
        kpi(col, f"{STAGE_META[s][0]} {s}", cnt[s], "", "#2563eb")
    st.caption(f"❌ Payment failed: {cnt['Payment Failed']}   •   🚫 Cancelled: {cnt['Cancelled']}")

    a, b = st.columns([1.4, 1])
    with a:
        st.markdown('<div class="sec">Recent orders</div>', unsafe_allow_html=True)
        st.dataframe(orders_df(orders[:7]), use_container_width=True, hide_index=True, column_config=TABLE_CFG)
    with b:
        st.markdown('<div class="sec">Orders by stage</div>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Orders": cnt.values()}, index=list(cnt)), use_container_width=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="sec">Revenue by category</div>', unsafe_allow_html=True)
        cat = pd.DataFrame(alive).groupby("category")["total"].sum() if alive else pd.Series(dtype=float)
        st.bar_chart(cat, use_container_width=True)
    with d2:
        st.markdown('<div class="sec">Payment methods</div>', unsafe_allow_html=True)
        st.bar_chart(pd.Series([o["payment_method"] for o in orders]).value_counts(), use_container_width=True)
    with d3:
        st.markdown('<div class="sec">Orders per day</div>', unsafe_allow_html=True)
        st.line_chart(pd.Series([o["created"].date() for o in orders]).value_counts().sort_index(), use_container_width=True)
    if low:
        st.warning("Low-stock alert: " + ", ".join(p["name"] for p in low))


# --------------------------------------------------------------
# 7. PLACE ORDER
# --------------------------------------------------------------

elif page == "Place Order":
    header("Place a New Order", "Choose a product, pay and watch the order enter the fulfilment pipeline.")
    avail = {f"{p['emoji']} {p['name']} - {money(p['price'])} ({p['stock']} left)": p for p in products if p["stock"] > 0}
    if not avail:
        st.warning("Everything is out of stock. Restock from the Inventory page.")
    else:
        L, R = st.columns([1.2, 1])
        with L, st.container(border=True):
            customer = st.text_input("Customer name", "Guest Customer")
            c1, c2 = st.columns(2)
            prod = avail[c1.selectbox("Product", list(avail))]
            qty = c2.number_input("Quantity", 1, 20, 1)
            c3, c4 = st.columns(2)
            city = c3.selectbox("Delivery city", list(CITIES))
            method = c4.selectbox("Payment method", PAY_ALL)
            coupon = st.text_input("Coupon code (try WELCOME10, SAVE5, FESTIVE15)")
        q = quote(prod["price"], int(qty), coupon)
        with R:
            H(f'<div class="card"><b>Price summary</b><br><br>Subtotal: {money(q["subtotal"])}<br>'
              f'Coupon ({q["pct"]}%): -{money(q["discount"])}<br>GST (18%): {money(q["gst"])}<br>'
              f'Shipping: {"FREE" if q["shipping"] == 0 else money(q["shipping"])}<hr>'
              f'<span style="font-size:24px;font-weight:800">{money(q["total"])}</span><br>'
              f'<small>Estimated delivery: {CITIES[city]} days after shipping</small></div>')
            if coupon and not q["pct"]:
                st.caption("Coupon not valid.")
        if st.button("🛒 Place order", type="primary", use_container_width=True):
            ok, msg, o = create_order(store, prod["id"], int(qty), method, city, coupon,
                                      html.escape(customer.strip() or "Guest"), datetime.now())
            if ok:
                st.session_state["track_id"] = o["id"]
                st.success(msg + " Open **Order Tracker** to process it through payment, packing, shipping and delivery.")
                st.balloons()
            else:
                st.error(msg)


# --------------------------------------------------------------
# 8. ORDER TRACKER (+ all orders)
# --------------------------------------------------------------

elif page == "Order Tracker":
    header("Order Tracker", "Follow every order from placement to doorstep and process it stage by stage.")
    tab1, tab2 = st.tabs(["🔎 Track & process", "📋 All orders"])

    with tab1:
        ids = [o["id"] for o in orders]
        if st.session_state.get("track_id") not in ids:
            st.session_state.pop("track_id", None)
        oid = st.selectbox("Select order", ids, key="track_id",
                           format_func=lambda i: next(f"#{o['id']} · {o['emoji']} {o['product']} · {o['status']}"
                                                      for o in orders if o["id"] == i))
        o = next(x for x in orders if x["id"] == oid)
        ph_sum, ph_step = st.empty(), st.empty()
        active = o["status"] in FLOW and o["status"] != "Delivered"

        def draw():
            with ph_sum.container():
                k = st.columns(4)
                kpi(k[0], "Status", f"{ICON[o['status']]} {o['status']}", f"{o['customer']} • {o['city']}")
                kpi(k[1], "Payment", o["payment_status"], o["payment_method"], "#f79009")
                kpi(k[2], "Courier", o["courier"], f"Tracking: {o['tracking']}", "#7c3aed")
                kpi(k[3], "Order total", money(o["total"]),
                    f"ETA: {o['eta'].strftime('%d %b') if o['eta'] else 'after shipping'}", "#12b76a")
            ph_step.markdown(stepper(o), unsafe_allow_html=True)

        b = st.columns(4)
        if b[0].button("Move to next stage", type="primary", disabled=not active, use_container_width=True):
            ok, m = advance(store, o, next_time(o), fail_pct / 100)
            st.toast(m, icon="✅" if ok else "⚠️")
        if b[1].button("▶ Auto-play journey", disabled=not active, use_container_width=True):
            while o["status"] in FLOW and o["status"] != "Delivered":
                ok, m = advance(store, o, next_time(o), fail_pct / 100)
                draw()
                if not ok:
                    st.toast(m, icon="⚠️")
                    break
                time.sleep(0.9)
        if b[2].button("Cancel order", disabled=o["status"] not in ("Placed", "Payment Verified", "Packed"),
                       use_container_width=True):
            st.toast(cancel_order(store, o, next_time(o))[1], icon="🚫")
        if b[3].button("Retry payment", disabled=o["status"] != "Payment Failed", use_container_width=True):
            st.toast(retry_payment(store, o, next_time(o))[1], icon="💳")
        draw()

        if active:
            nxt = FLOW[min(FLOW.index(o["status"]) + 1, len(FLOW) - 1)]
            m = STAGE_META[nxt]
            H(f'<div class="note"><b>Next: {m[0]} {nxt}</b><br>Customer sees: {m[1]}<br>System does: {m[2]}</div>')
        elif o["status"] == "Payment Failed":
            st.error("Payment failed. Stock was released. The customer can retry the payment.")
        elif o["status"] == "Delivered":
            st.success("Order delivered. The journey is complete.")

        st.markdown('<div class="sec">Order timeline (simulated clock)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Time": e["time"].strftime("%d %b, %I:%M %p"), "Who": e["actor"],
                                    "What happened": e["event"]} for e in o["events"]]),
                     use_container_width=True, hide_index=True)
        with st.expander("🧾 Invoice"):
            st.write(f"**Invoice for order #{o['id']}** - {o['customer']}, {o['city']}")
            st.write(f"{o['qty']} x {o['product']}: {money(o['subtotal'])}  |  Discount ({o['coupon'] or 'none'}): "
                     f"-{money(o['discount'])}  |  GST: {money(o['gst'])}  |  Shipping: {money(o['shipping'])}")
            st.write(f"**Total: {money(o['total'])}**")

    with tab2:
        f1, f2 = st.columns([2, 1])
        stat = f1.multiselect("Filter by status", list(STAGE_META) + list(BAD))
        text = f2.text_input("Search customer / product")
        shown = [o for o in orders if (not stat or o["status"] in stat)
                 and text.lower() in f"{o['customer']} {o['product']}".lower()]
        df = orders_df(shown)
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=TABLE_CFG)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", df.to_csv(index=False), "orders.csv", "text/csv", use_container_width=True)
        if d2.button("⏩ Move all active orders one stage", use_container_width=True):
            moved = sum(advance(store, o, next_time(o), fail_pct / 100)[0]
                        for o in orders if o["status"] in FLOW and o["status"] != "Delivered")
            st.toast(f"{moved} orders moved forward.")
            st.rerun()


# --------------------------------------------------------------
# 9. INVENTORY
# --------------------------------------------------------------

elif page == "Inventory":
    header("Inventory Management", "Products, stock levels, low-stock alerts and restocking.")
    c = st.columns(3)
    kpi(c[0], "Products", len(products), "Listed SKUs")
    kpi(c[1], "Units in stock", sum(p["stock"] for p in products), "In warehouse", "#12b76a")
    kpi(c[2], "Stock value", money(sum(p["stock"] * p["price"] for p in products)), "Stock x price", "#7c3aed")
    st.write("")
    st.dataframe(pd.DataFrame([{
        "Product": f"{p['emoji']} {p['name']}", "ID": p["id"], "Category": p["category"], "Price": p["price"],
        "Stock": p["stock"], "Sold": p["sold"], "Status": "⚠ Low stock" if p["stock"] <= 5 else "✓ In stock"}
        for p in products]), use_container_width=True, hide_index=True,
        column_config={"Price": st.column_config.NumberColumn(format="₹%d"),
                       "Stock": st.column_config.ProgressColumn(min_value=0, max_value=50, format="%d")})
    if low:
        st.warning("Low-stock alert: " + ", ".join(p["name"] for p in low))
    st.bar_chart(pd.DataFrame({"Stock": [p["stock"] for p in products]}, index=[p["name"] for p in products]))

    a, r = st.columns(2)
    with a, st.form("add_product"):
        st.markdown("**Add a product**")
        name = st.text_input("Product name")
        cat = st.selectbox("Category", ["Electronics", "Accessories", "Wearables", "Other"])
        price = st.number_input("Selling price (₹)", 1, 500000, 1000, 100)
        stock = st.number_input("Opening stock", 0, 1000, 10)
        if st.form_submit_button("Add product"):
            if not name.strip():
                st.error("Enter a product name.")
            elif any(p["name"].lower() == name.strip().lower() for p in products):
                st.error("This product already exists.")
            else:
                nid = max(int(p["id"][1:]) for p in products) + 1
                products.append({"id": f"P{nid}", "name": name.strip(), "emoji": "🛍️", "category": cat,
                                 "price": int(price), "stock": int(stock), "sold": 0})
                st.rerun()
    with r, st.form("restock"):
        st.markdown("**Restock a product**")
        pick = st.selectbox("Product", [p["id"] for p in products],
                            format_func=lambda i: f"{get_product(store, i)['name']} (stock {get_product(store, i)['stock']})")
        add = st.number_input("Units to add", 1, 1000, 10)
        if st.form_submit_button("＋ Update stock"):
            get_product(store, pick)["stock"] += int(add)
            st.rerun()


# --------------------------------------------------------------
# 10. SIMULATION LAB
# --------------------------------------------------------------

elif page == "Simulation Lab":
    header("Simulation Lab", "Run hour-by-hour what-if experiments on a sandbox copy of the store.")
    H('<div class="note"><b>Model:</b> every hour a random number of customers arrive (Poisson). Each order picks a product '
      '(popular ones more often), pays (can fail), gets packed, shipped and delivered with random delays. '
      'Orders may be cancelled and out-of-stock demand is a lost sale.</div>')
    st.write("")
    with st.container(border=True):
        a, b, c = st.columns(3)
        hours = a.slider("Simulation length (hours)", 6, 168, 48)
        rate = a.slider("Customer arrivals per hour", 0.5, 15.0, 3.0, 0.5)
        fail = b.slider("Payment failure (%)", 0, 40, fail_pct)
        cod = b.slider("Cash on Delivery share (%)", 0, 80, 30)
        cancel_pct = c.slider("Cancellation (%)", 0, 30, 5)
        restock = c.checkbox("Auto-restock low items", True)
        thr = c.number_input("Restock when stock ≤", 1, 20, 5)
        seed = a.number_input("Random seed (0 = random)", 0, 9999, 0)
        if st.button("▶ Run simulation", type="primary", use_container_width=True):
            if seed:
                random.seed(int(seed))
                np.random.seed(int(seed))
            with st.spinner("Simulating..."):
                st.session_state.sim = simulate(copy.deepcopy(store), hours, rate, fail, cod, cancel_pct,
                                                restock, int(thr), datetime.now() - timedelta(hours=hours))

    sim = st.session_state.get("sim")
    if sim:
        st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)
        r1 = st.columns(5)
        kpi(r1[0], "Customers arrived", sim["arrived"], "Orders attempted")
        kpi(r1[1], "Orders accepted", sim["accepted"], "Stock was available", "#12b76a")
        kpi(r1[2], "Delivered", sim["delivered"], "Completed in time window", "#12b76a")
        kpi(r1[3], "Payment failed", sim["failed"], "Gateway declined", "#f04438")
        kpi(r1[4], "Cancelled", sim["cancelled"], "By customer", "#f79009")
        r2 = st.columns(5)
        kpi(r2[0], "Revenue", money(sim["revenue"]), "Valid orders", "#12b76a")
        kpi(r2[1], "Avg delivery time", f"{sim['avg_hours']:.0f} hrs" if sim["avg_hours"] else "-", "Order to doorstep", "#7c3aed")
        kpi(r2[2], "Fill rate", f"{sim['accepted'] / sim['arrived'] * 100:.0f}%" if sim["arrived"] else "-", "Accepted / arrived")
        kpi(r2[3], "Lost sales", sim["lost"], "Out of stock", "#f04438")
        kpi(r2[4], "Auto-restocks", sim["restocks"], "Restock events", "#f79009")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Orders in pipeline vs delivered**")
            st.line_chart(sim["series"][["In pipeline", "Delivered (cumulative)"]])
        with g2:
            st.markdown("**Total units in stock over time**")
            st.line_chart(sim["series"][["Units in stock"]])
        st.markdown("**Final order stages (simulated orders)**")
        st.bar_chart(pd.DataFrame({"Orders": sim["stages"]}, index=list(STAGE_META) + list(BAD)))
        if st.button("Apply this simulation to the live dashboard"):
            st.session_state.store = sim["store"]
            st.session_state.pop("sim")
            st.rerun()


# --------------------------------------------------------------
# 11. HOW IT WORKS
# --------------------------------------------------------------

elif page == "How It Works":
    header("How E-Commerce Works", "What happens after the customer clicks 'Place order' - step by step.")
    st.graphviz_chart("""digraph { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#eaf2ff", color="#2563eb", fontname="Helvetica"];
      Browse -> Cart -> Checkout -> Placed -> "Payment Gateway";
      "Payment Gateway" -> "Payment Verified" [label="success", color="#12b76a"];
      "Payment Gateway" -> "Payment Failed" [label="declined", color="#f04438"];
      "Payment Failed" [fillcolor="#fee4e2", color="#f04438"];
      "Payment Failed" -> Placed [label="retry", style=dashed];
      "Payment Verified" -> Packed -> Shipped -> "In Transit" -> "Out for Delivery" -> Delivered;
      Placed -> Cancelled [style=dashed]; Packed -> Cancelled [style=dashed];
      Cancelled [fillcolor="#fef0c7", color="#f79009"]; Delivered [fillcolor="#d1fadf", color="#12b76a"]; }""")
    for s, (icon, sees, does) in STAGE_META.items():
        H(f'<div class="flow"><b>{icon} {s}</b><span><b>Customer sees:</b> {sees}</span><span><b>System does:</b> {does}</span></div>')
    x, y = st.columns(2)
    with x:
        st.markdown("**Price formula**")
        st.code("Total = (Subtotal - Coupon) + 18% GST + Shipping\nShipping = ₹0 if amount ≥ ₹999 else ₹49")
        st.markdown("**Refunds & failures**")
        st.write("Failed payment releases the reserved stock. Cancelling a paid order before shipping starts a refund.")
    with y:
        st.markdown("**Modelling concepts used in this project**")
        st.write("• **Entities:** customers, orders, products\n\n• **State:** order status, payment status, stock\n\n"
                 "• **Events:** order, pay, pack, ship, deliver, cancel\n\n• **Randomness:** Poisson arrivals, payment failure, delays\n\n"
                 "• **Output metrics:** fill rate, delivery time, revenue, lost sales")

st.markdown('<div style="color:#98a2b3;font-size:11px;text-align:center;padding:22px 0 4px">ShopFlow • E-Commerce Order Processing Simulation • NEC00091 • Built with Python and Streamlit</div>',
            unsafe_allow_html=True)
