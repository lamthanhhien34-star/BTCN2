
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ============================================================

st.set_page_config(
    page_title="Amazon Baby Product Intelligence",
    page_icon="🧸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. GIAO DIỆN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(124,58,237,.08), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        padding: 26px 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #4338ca 0%, #7c3aed 55%, #a855f7 100%);
        color: white;
        box-shadow: 0 16px 38px rgba(79, 70, 229, .18);
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        font-weight: 850;
    }

    .hero p {
        margin: 8px 0 0 0;
        font-size: 1rem;
        opacity: .94;
    }

    .kpi {
        background: rgba(255,255,255,.96);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 116px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, .06);
    }

    .kpi-label {
        color: #64748b;
        font-size: .77rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 850;
        margin-top: 6px;
    }

    .kpi-note {
        color: #64748b;
        font-size: .80rem;
        margin-top: 4px;
    }

    .insight {
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #6366f1;
        border-radius: 15px;
        padding: 14px 16px;
        margin: 12px 0 16px 0;
        box-shadow: 0 6px 16px rgba(15,23,42,.04);
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 5px solid #f97316;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0 16px 0;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 850;
        color: #111827;
        margin: 8px 0 8px 0;
    }

    .product-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 8px 22px rgba(15,23,42,.05);
        margin-bottom: 10px;
    }

    .badge-high {
        background:#fee2e2;
        color:#991b1b;
        border-radius:999px;
        padding:4px 10px;
        font-weight:700;
        font-size:.78rem;
    }

    .badge-medium {
        background:#fef3c7;
        color:#92400e;
        border-radius:999px;
        padding:4px 10px;
        font-weight:700;
        font-size:.78rem;
    }

    .badge-low {
        background:#dcfce7;
        color:#166534;
        border-radius:999px;
        padding:4px 10px;
        font-weight:700;
        font-size:.78rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 12px 14px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, .04);
    }

    .footer {
        text-align:center;
        color:#64748b;
        font-size:.8rem;
        padding:22px 0 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. TỰ ĐỘNG TÌM THƯ MỤC DỮ LIỆU
# ============================================================

ROOT = Path(__file__).resolve().parent

def tim_thu_muc_du_lieu():
    thu_muc_uu_tien = [
        ROOT / "data",
        ROOT / "web_bundle",
        ROOT / "amazon_baby_web_bundle",
        ROOT,
    ]
    for folder in thu_muc_uu_tien:
        if (folder / "overview.json").exists():
            return folder

    found = list(ROOT.rglob("overview.json"))
    if found:
        return found[0].parent

    return ROOT / "data"

DATA_DIR = tim_thu_muc_du_lieu()


# ============================================================
# 4. HÀM HỖ TRỢ
# ============================================================

@st.cache_data(show_spinner=False)
def doc_json(ten_file):
    path = DATA_DIR / ten_file
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def doc_csv(ten_file):
    path = DATA_DIR / ten_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def doc_parquet(ten_file):
    path = DATA_DIR / ten_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def doi_ngay(df, cot):
    if not df.empty and cot in df.columns:
        df = df.copy()
        df[cot] = pd.to_datetime(df[cot], errors="coerce")
    return df

def fmt_int(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{int(round(float(x))):,}"

def fmt_pct(x, n=1):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x)*100:.{n}f}%"

def fmt_num(x, n=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.{n}f}"

def fmt_money(x):
    if x is None or pd.isna(x):
        return "—"
    return f"${float(x):,.2f}"

def show_kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

ASPECT_VI = {
    "Quality & Durability": "Chất lượng & độ bền",
    "Safety-related": "Liên quan đến an toàn",
    "Ease of Use": "Dễ sử dụng",
    "Comfort": "Sự thoải mái",
    "Cleaning": "Vệ sinh",
    "Leakage & Sealing": "Rò rỉ & độ kín",
    "Material & Odor": "Chất liệu & mùi",
    "Size & Fit": "Kích thước & độ vừa vặn",
    "Price & Value": "Giá & giá trị",
    "Delivery & Packaging": "Giao hàng & đóng gói",
}

RISK_VI = {
    "Critical": "Rất cao",
    "High": "Cao",
    "Medium": "Trung bình",
    "Low": "Thấp",
}

SIGNAL_VI = {
    "Critical": "Rất cao",
    "High": "Cao",
    "Medium": "Trung bình",
    "Low": "Thấp",
}

SENTIMENT_VI = {
    "Positive": "Tích cực",
    "Neutral": "Trung lập",
    "Negative": "Tiêu cực",
}

EXPERIENCE_VI = {
    "Mostly confirmed": "Phần lớn được xác nhận",
    "Mixed": "Trải nghiệm trái chiều",
    "Frequently contradicted": "Thường xuyên bị phản bác",
    "Insufficient evidence": "Chưa đủ bằng chứng",
}


# ============================================================
# 5. ĐỌC DỮ LIỆU
# ============================================================

overview = doc_json("overview.json")
pipeline = doc_json("pipeline_metrics.json")
nlp_validation = doc_json("nlp_validation.json")

product_summary = doc_parquet("product_summary.parquet")
monthly_trend = doi_ngay(doc_parquet("monthly_trend.parquet"), "review_month")
subcategory_summary = doc_parquet("subcategory_summary.parquet")
product_monthly = doi_ngay(doc_parquet("product_monthly.parquet"), "review_month")
aspect_summary = doc_parquet("aspect_summary.parquet")
aspect_product_summary = doc_parquet("aspect_product_summary.parquet")
reviews = doi_ngay(doc_parquet("representative_reviews.parquet"), "review_date")
complaint_monthly = doi_ngay(doc_parquet("complaint_monthly.parquet"), "review_month")
emerging_issues = doi_ngay(doc_parquet("emerging_issues.parquet"), "review_month")
risk_scores = doc_parquet("risk_scores.parquet")
claim_experience = doc_parquet("claim_experience.parquet")
nlp_sample = doi_ngay(doc_parquet("nlp_sentiment_sample.parquet"), "review_date")
spark_benchmark = doc_csv("spark_benchmark.csv")
data_quality = doc_csv("data_quality.csv")


# ============================================================
# 6. KIỂM TRA FILE
# ============================================================

required_files = [
    "overview.json",
    "pipeline_metrics.json",
    "product_summary.parquet",
    "monthly_trend.parquet",
    "subcategory_summary.parquet",
    "product_monthly.parquet",
    "aspect_summary.parquet",
    "aspect_product_summary.parquet",
    "representative_reviews.parquet",
    "complaint_monthly.parquet",
    "emerging_issues.parquet",
    "risk_scores.parquet",
    "claim_experience.parquet",
    "nlp_sentiment_sample.parquet",
    "nlp_validation.json",
    "spark_benchmark.csv",
    "data_quality.csv",
]

missing = [f for f in required_files if not (DATA_DIR / f).exists()]

if missing:
    st.error("Ứng dụng chưa tìm thấy đầy đủ dữ liệu kết quả.")
    with st.expander("Xem chi tiết"):
        st.write("Thư mục dữ liệu đang dùng:", str(DATA_DIR))
        st.write("Các file còn thiếu:")
        for f in missing:
            st.code(f)
    st.stop()


# ============================================================
# 7. LOOKUP SẢN PHẨM
# ============================================================

lookup_cols = [
    c for c in [
        "parent_asin",
        "product_title",
        "store",
        "subcategory",
        "price",
        "avg_rating",
        "reviews",
        "negative_rate",
        "verified_rate",
    ]
    if c in product_summary.columns
]

product_lookup = product_summary[lookup_cols].drop_duplicates("parent_asin").copy()

def merge_product_info(df):
    if df.empty or "parent_asin" not in df.columns:
        return df.copy()

    base = df.copy()
    cols_available = [
        c for c in ["parent_asin", "product_title", "store", "subcategory", "price"]
        if c in product_lookup.columns
    ]

    already = [c for c in cols_available if c != "parent_asin" and c in base.columns]
    needed = [c for c in cols_available if c == "parent_asin" or c not in already]

    if len(needed) <= 1:
        return base

    return base.merge(
        product_lookup[needed],
        on="parent_asin",
        how="left"
    )

risk_full = merge_product_info(risk_scores)
emerging_full = merge_product_info(emerging_issues)
aspect_product_full = merge_product_info(aspect_product_summary)
complaint_full = merge_product_info(complaint_monthly)
claim_full = merge_product_info(claim_experience)


# ============================================================
# 8. THANH BÊN
# ============================================================

with st.sidebar:
    st.markdown("## 🧸 BABY PRODUCT INTELLIGENCE")
    st.caption("Phân tích tiếng nói khách hàng trên Amazon")

    page = st.radio(
        "Chọn nội dung",
        [
            "1. Tổng quan",
            "2. Sản phẩm cần chú ý",
            "3. Chi tiết sản phẩm",
            "4. Tiếng nói khách hàng",
            "5. Big Data & phương pháp",
        ],
    )

    st.markdown("---")
    st.markdown("### Bộ lọc")

    categories = []
    if not product_summary.empty and "subcategory" in product_summary.columns:
        categories = sorted(
            product_summary["subcategory"].dropna().astype(str).unique().tolist()
        )

    selected_categories = st.multiselect(
        "Danh mục con",
        categories,
        placeholder="Tất cả danh mục",
    )

    min_reviews = st.slider(
        "Số review tối thiểu của sản phẩm",
        min_value=0,
        max_value=500,
        value=30,
        step=10,
    )

    st.markdown("---")
    st.caption(
        f"Chế độ phân tích: {pipeline.get('analysis_mode', overview.get('analysis_mode', 'FAST'))}"
    )
    st.caption(
        f"Dữ liệu phân tích: {fmt_int(pipeline.get('analysis_reviews', overview.get('reviews', 0)))} review"
    )


# ============================================================
# 9. LỌC TOÀN CỤC
# ============================================================

product_filtered = product_summary.copy()

if selected_categories and "subcategory" in product_filtered.columns:
    product_filtered = product_filtered[
        product_filtered["subcategory"].isin(selected_categories)
    ]

if "reviews" in product_filtered.columns:
    product_filtered = product_filtered[
        product_filtered["reviews"] >= min_reviews
    ]

selected_asins = set()
if not product_filtered.empty and "parent_asin" in product_filtered.columns:
    selected_asins = set(product_filtered["parent_asin"].astype(str))


# ============================================================
# PAGE 1 - TỔNG QUAN
# ============================================================

if page == "1. Tổng quan":

    hero(
        "Tổng quan trải nghiệm khách hàng",
        "Theo dõi nhanh tình hình đánh giá, sản phẩm cần chú ý và các chủ đề phản hồi đang thay đổi.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        show_kpi("Review đã phân tích", fmt_int(overview.get("reviews")), "Sau làm sạch")
    with c2:
        show_kpi("Số sản phẩm", fmt_int(overview.get("products")), "Toàn bộ tập phân tích")
    with c3:
        show_kpi("Điểm trung bình", f"{fmt_num(overview.get('avg_rating'))} ★", "Thang điểm 1–5")
    with c4:
        show_kpi("Tỷ lệ tiêu cực", fmt_pct(overview.get("negative_rate")), "Review 1–2 sao")
    with c5:
        show_kpi("Mua hàng xác minh", fmt_pct(overview.get("verified_rate")), "Verified Purchase")

    # Nhận xét tự động
    top_aspect = None
    if not aspect_summary.empty:
        tmp = aspect_summary.sort_values(
            ["negative_rate", "mentions"],
            ascending=[False, False]
        )
        if not tmp.empty:
            top_aspect = ASPECT_VI.get(tmp.iloc[0]["aspect"], tmp.iloc[0]["aspect"])

    high_risk_count = 0
    if not risk_full.empty and "risk_level" in risk_full.columns:
        high_risk_count = int(risk_full["risk_level"].isin(["Critical", "High"]).sum())

    st.markdown(
        f"""
        <div class="insight">
            <b>Điểm cần chú ý:</b>
            dữ liệu hiện có tỷ lệ review tiêu cực là <b>{fmt_pct(overview.get('negative_rate'))}</b>.
            Có <b>{high_risk_count}</b> sản phẩm thuộc nhóm ưu tiên cao hoặc rất cao.
            Chủ đề cần theo dõi nổi bật hiện tại là
            <b>{top_aspect if top_aspect else 'chưa đủ dữ liệu xác định'}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Top sản phẩm cần chú ý</div>', unsafe_allow_html=True)

    if not risk_full.empty:
        top_risk = risk_full.copy()

        if selected_categories and "subcategory" in top_risk.columns:
            top_risk = top_risk[top_risk["subcategory"].isin(selected_categories)]

        if "reviews" in top_risk.columns:
            top_risk = top_risk[top_risk["reviews"] >= min_reviews]

        top_risk = top_risk.sort_values("risk_score", ascending=False).head(10)

        show_cols = [
            c for c in [
                "product_title",
                "store",
                "subcategory",
                "reviews",
                "avg_rating",
                "negative_rate",
                "risk_score",
                "risk_level",
            ]
            if c in top_risk.columns
        ]

        display_df = top_risk[show_cols].copy()

        rename_map = {
            "product_title": "Sản phẩm",
            "store": "Cửa hàng / thương hiệu",
            "subcategory": "Danh mục",
            "reviews": "Số review",
            "avg_rating": "Điểm TB",
            "negative_rate": "Tỷ lệ tiêu cực",
            "risk_score": "Điểm ưu tiên",
            "risk_level": "Mức ưu tiên",
        }

        display_df = display_df.rename(columns=rename_map)

        if "Mức ưu tiên" in display_df.columns:
            display_df["Mức ưu tiên"] = display_df["Mức ưu tiên"].map(RISK_VI).fillna(display_df["Mức ưu tiên"])

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tỷ lệ tiêu cực": st.column_config.ProgressColumn(
                    "Tỷ lệ tiêu cực",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1f%%",
                ),
                "Điểm ưu tiên": st.column_config.ProgressColumn(
                    "Điểm ưu tiên",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
                ),
            }
        )
    else:
        st.info("Chưa có dữ liệu điểm ưu tiên sản phẩm.")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="section-title">Xu hướng review theo thời gian</div>', unsafe_allow_html=True)

        if not monthly_trend.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=monthly_trend["review_month"],
                    y=monthly_trend["reviews"],
                    name="Số review",
                    marker_color="#6366f1",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly_trend["review_month"],
                    y=monthly_trend["negative_rate"],
                    name="Tỷ lệ tiêu cực",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=3),
                )
            )

            fig.update_layout(
                height=410,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(title="Số review"),
                yaxis2=dict(
                    title="Tỷ lệ tiêu cực",
                    overlaying="y",
                    side="right",
                    tickformat=".0%",
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Khách hàng đang phàn nàn về gì?</div>', unsafe_allow_html=True)

        if not aspect_summary.empty:
            tmp = aspect_summary.copy()
            tmp["Khía cạnh"] = tmp["aspect"].map(ASPECT_VI).fillna(tmp["aspect"])
            tmp = tmp.sort_values("negative_rate", ascending=True)

            fig = px.bar(
                tmp,
                x="negative_rate",
                y="Khía cạnh",
                orientation="h",
                color="mentions",
                labels={
                    "negative_rate": "Tỷ lệ tiêu cực",
                    "mentions": "Lượt đề cập",
                },
            )

            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(height=410, coloraxis_showscale=False)

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2 - SẢN PHẨM CẦN CHÚ Ý
# ============================================================

elif page == "2. Sản phẩm cần chú ý":

    hero(
        "Sản phẩm cần chú ý",
        "Xác định sản phẩm nào cần được kiểm tra trước và lý do vì sao sản phẩm đó được xếp mức ưu tiên cao.",
    )

    st.markdown(
        """
        <div class="warning-box">
            <b>Lưu ý:</b> Điểm ưu tiên là tín hiệu phân tích dựa trên review khách hàng,
            không phải chứng nhận an toàn hay kết luận chất lượng sản phẩm.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if risk_full.empty:
        st.info("Chưa có dữ liệu điểm ưu tiên.")
    else:
        temp = risk_full.copy()

        if selected_categories and "subcategory" in temp.columns:
            temp = temp[temp["subcategory"].isin(selected_categories)]

        if "reviews" in temp.columns:
            temp = temp[temp["reviews"] >= min_reviews]

        if temp.empty:
            st.info("Không có sản phẩm phù hợp với bộ lọc.")
        else:
            level_options = ["Tất cả", "Rất cao", "Cao", "Trung bình", "Thấp"]
            level_selected = st.selectbox("Lọc mức ưu tiên", level_options)

            if level_selected != "Tất cả":
                reverse_risk = {v: k for k, v in RISK_VI.items()}
                temp = temp[temp["risk_level"] == reverse_risk[level_selected]]

            search_text = st.text_input(
                "🔎 Tìm theo tên sản phẩm / thương hiệu",
                placeholder="Nhập tên sản phẩm hoặc store..."
            )

            if search_text.strip():
                q = search_text.strip().lower()
                mask = (
                    temp["product_title"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
                    |
                    temp["store"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
                )
                temp = temp[mask]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sản phẩm trong bộ lọc", fmt_int(len(temp)))
            c2.metric("Ưu tiên rất cao", fmt_int((temp["risk_level"] == "Critical").sum()))
            c3.metric("Ưu tiên cao", fmt_int((temp["risk_level"] == "High").sum()))
            c4.metric("Điểm ưu tiên trung vị", fmt_num(temp["risk_score"].median(), 1))

            # Biểu đồ top sản phẩm
            top = temp.sort_values("risk_score", ascending=False).head(20).copy()
            top["Tên ngắn"] = top["product_title"].fillna("Không rõ tên").astype(str).str.slice(0, 60)

            fig = px.bar(
                top.sort_values("risk_score"),
                x="risk_score",
                y="Tên ngắn",
                orientation="h",
                color="risk_level",
                color_discrete_map={
                    "Critical": "#dc2626",
                    "High": "#f97316",
                    "Medium": "#eab308",
                    "Low": "#16a34a",
                },
                labels={
                    "risk_score": "Điểm ưu tiên",
                    "Tên ngắn": "",
                    "risk_level": "Mức độ",
                },
                hover_data=["store", "subcategory", "reviews", "avg_rating", "negative_rate"],
            )

            fig.update_layout(height=680)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Vì sao các sản phẩm này cần chú ý?")

            explain_cols = [
                c for c in [
                    "product_title",
                    "store",
                    "reviews",
                    "avg_rating",
                    "negative_rate",
                    "max_signal_score",
                    "helpful_votes",
                    "risk_score",
                    "risk_level",
                ]
                if c in temp.columns
            ]

            explain = temp.sort_values("risk_score", ascending=False).head(30)[explain_cols].copy()

            explain = explain.rename(columns={
                "product_title": "Sản phẩm",
                "store": "Cửa hàng / thương hiệu",
                "reviews": "Số review",
                "avg_rating": "Điểm TB",
                "negative_rate": "Tỷ lệ tiêu cực",
                "max_signal_score": "Tín hiệu tăng bất thường",
                "helpful_votes": "Helpful votes",
                "risk_score": "Điểm ưu tiên",
                "risk_level": "Mức ưu tiên",
            })

            if "Mức ưu tiên" in explain.columns:
                explain["Mức ưu tiên"] = explain["Mức ưu tiên"].map(RISK_VI).fillna(explain["Mức ưu tiên"])

            st.dataframe(explain, use_container_width=True, hide_index=True)

            st.caption(
                "Điểm ưu tiên hiện tại được xây dựng từ tỷ lệ review tiêu cực, "
                "tín hiệu phản hồi tăng bất thường, thành phần rating thấp và helpful votes."
            )


# ============================================================
# PAGE 3 - CHI TIẾT SẢN PHẨM
# ============================================================

elif page == "3. Chi tiết sản phẩm":

    hero(
        "Chi tiết sản phẩm",
        "Theo dõi một sản phẩm cụ thể: hiệu suất đánh giá, vấn đề nổi bật, xu hướng theo thời gian và review bằng chứng.",
    )

    if product_filtered.empty:
        st.info("Không có sản phẩm phù hợp với bộ lọc.")
    else:
        product_choices = product_filtered.sort_values("reviews", ascending=False).head(500).copy()

        search = st.text_input(
            "🔎 Tìm sản phẩm",
            placeholder="Nhập tên sản phẩm, store hoặc Parent ASIN"
        )

        if search.strip():
            q = search.strip().lower()

            mask = (
                product_choices["product_title"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
                |
                product_choices["store"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
                |
                product_choices["parent_asin"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
            )

            product_choices = product_choices[mask]

        if product_choices.empty:
            st.info("Không tìm thấy sản phẩm.")
        else:
            product_choices["display_name"] = (
                product_choices["product_title"].fillna("Không rõ tên").astype(str).str.slice(0, 95)
                + " · "
                + product_choices["store"].fillna("Không rõ store").astype(str).str.slice(0, 30)
            )

            selected_name = st.selectbox(
                "Chọn sản phẩm",
                product_choices["display_name"].tolist()
            )

            row = product_choices[product_choices["display_name"] == selected_name].iloc[0]
            asin = str(row["parent_asin"])

            st.markdown(f"## {row.get('product_title', 'Không rõ tên sản phẩm')}")
            st.caption(
                f"Cửa hàng / thương hiệu: {row.get('store', 'Không rõ')} · "
                f"Danh mục: {row.get('subcategory', 'Không rõ')} · "
                f"Parent ASIN: {asin}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Điểm trung bình", f"{fmt_num(row.get('avg_rating'))} ★")
            c2.metric("Số review", fmt_int(row.get("reviews")))
            c3.metric("Tỷ lệ tiêu cực", fmt_pct(row.get("negative_rate")))
            c4.metric("Verified Purchase", fmt_pct(row.get("verified_rate")))
            c5.metric("Giá", fmt_money(row.get("price")))

            # Risk info
            risk_row = pd.DataFrame()
            if not risk_full.empty:
                risk_row = risk_full[risk_full["parent_asin"].astype(str) == asin]

            if not risk_row.empty:
                rr = risk_row.iloc[0]
                st.markdown(
                    f"""
                    <div class="insight">
                        <b>Mức ưu tiên hiện tại:</b> {RISK_VI.get(rr.get('risk_level'), rr.get('risk_level', 'Không rõ'))}
                        · <b>Điểm:</b> {fmt_num(rr.get('risk_score'), 1)}/100.
                        Đây là chỉ số dùng để ưu tiên kiểm tra, không phải kết luận an toàn sản phẩm.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            left, right = st.columns([1.3, 1])

            with left:
                st.markdown('<div class="section-title">Xu hướng theo thời gian</div>', unsafe_allow_html=True)

                trend = product_monthly[
                    product_monthly["parent_asin"].astype(str) == asin
                ].copy() if not product_monthly.empty else pd.DataFrame()

                if not trend.empty:
                    trend = trend.sort_values("review_month")

                    fig = go.Figure()

                    fig.add_trace(
                        go.Bar(
                            x=trend["review_month"],
                            y=trend["reviews"],
                            name="Số review",
                            marker_color="#6366f1",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=trend["review_month"],
                            y=trend["negative_rate"],
                            name="Tỷ lệ tiêu cực",
                            yaxis="y2",
                            mode="lines+markers",
                            line=dict(color="#ef4444", width=3),
                        )
                    )

                    fig.update_layout(
                        height=420,
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.08),
                        yaxis2=dict(
                            title="Tỷ lệ tiêu cực",
                            overlaying="y",
                            side="right",
                            tickformat=".0%",
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Chưa có dữ liệu xu hướng theo tháng cho sản phẩm này.")

            with right:
                st.markdown('<div class="section-title">Khía cạnh cần chú ý</div>', unsafe_allow_html=True)

                asp = aspect_product_full[
                    aspect_product_full["parent_asin"].astype(str) == asin
                ].copy() if not aspect_product_full.empty else pd.DataFrame()

                if not asp.empty:
                    asp["Khía cạnh"] = asp["aspect"].map(ASPECT_VI).fillna(asp["aspect"])
                    asp = asp.sort_values("negative_rate", ascending=True)

                    fig = px.bar(
                        asp,
                        x="negative_rate",
                        y="Khía cạnh",
                        orientation="h",
                        color="mentions",
                        labels={
                            "negative_rate": "Tỷ lệ tiêu cực",
                            "mentions": "Lượt đề cập",
                        }
                    )
                    fig.update_xaxes(tickformat=".0%")
                    fig.update_layout(height=420, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Chưa có dữ liệu khía cạnh cho sản phẩm này.")

            # Emerging issues specific
            st.markdown("### Vấn đề đang tăng")

            em = emerging_full[
                emerging_full["parent_asin"].astype(str) == asin
            ].copy() if not emerging_full.empty else pd.DataFrame()

            if not em.empty:
                em["Khía cạnh"] = em["aspect"].map(ASPECT_VI).fillna(em["aspect"])
                em["Mức tín hiệu"] = em["signal_level"].map(SIGNAL_VI).fillna(em["signal_level"])

                show = em.sort_values("signal_score", ascending=False).head(10)[
                    [
                        c for c in [
                            "Khía cạnh",
                            "review_month",
                            "complaints",
                            "complaint_rate",
                            "growth_pct",
                            "z_score",
                            "signal_score",
                            "Mức tín hiệu",
                        ]
                        if c in em.columns
                    ]
                ].copy()

                show = show.rename(columns={
                    "review_month": "Tháng",
                    "complaints": "Số khiếu nại",
                    "complaint_rate": "Tỷ lệ khiếu nại",
                    "growth_pct": "Mức tăng",
                    "z_score": "Z-score",
                    "signal_score": "Điểm tín hiệu",
                })

                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa phát hiện tín hiệu tăng bất thường cho sản phẩm này.")

            # Reviews
            st.markdown("### Review bằng chứng")

            rv = reviews[
                reviews["parent_asin"].astype(str) == asin
            ].copy() if not reviews.empty else pd.DataFrame()

            if not rv.empty:
                sentiment_choice = st.selectbox(
                    "Lọc cảm xúc review",
                    ["Tất cả", "Tiêu cực", "Trung lập", "Tích cực"],
                )

                if sentiment_choice != "Tất cả":
                    reverse_sent = {v: k for k, v in SENTIMENT_VI.items()}
                    rv = rv[rv["rating_sentiment"] == reverse_sent[sentiment_choice]]

                rv = rv.sort_values("helpful_vote", ascending=False)

                for _, r in rv.head(15).iterrows():
                    sent_vi = SENTIMENT_VI.get(r.get("rating_sentiment"), r.get("rating_sentiment", ""))
                    with st.expander(
                        f"{int(round(r.get('rating', 0)))}★ · {sent_vi} · Helpful: {fmt_int(r.get('helpful_vote', 0))}"
                    ):
                        st.write(r.get("text", ""))
                        st.caption(
                            f"Verified Purchase: {'Có' if bool(r.get('verified_purchase', False)) else 'Không'}"
                        )
            else:
                st.info("Không có review đại diện cho sản phẩm này.")


# ============================================================
# PAGE 4 - TIẾNG NÓI KHÁCH HÀNG
# ============================================================

elif page == "4. Tiếng nói khách hàng":

    hero(
        "Tiếng nói khách hàng",
        "Khách hàng đang khen gì, phàn nàn gì và chủ đề nào đang tăng nhanh trong các review.",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Chủ đề nổi bật",
            "🔥 Vấn đề đang tăng",
            "🧠 Phân tích NLP",
        ]
    )

    with tab1:
        if aspect_summary.empty:
            st.info("Chưa có dữ liệu khía cạnh.")
        else:
            asp = aspect_summary.copy()
            asp["Khía cạnh"] = asp["aspect"].map(ASPECT_VI).fillna(asp["aspect"])

            c1, c2, c3 = st.columns(3)

            most_mentioned = asp.sort_values("mentions", ascending=False).iloc[0]
            most_negative = asp.sort_values("negative_rate", ascending=False).iloc[0]
            most_positive = asp.sort_values("positive_rate", ascending=False).iloc[0]

            c1.metric("Được nhắc nhiều nhất", most_mentioned["Khía cạnh"])
            c2.metric("Tiêu cực cao nhất", most_negative["Khía cạnh"])
            c3.metric("Tích cực cao nhất", most_positive["Khía cạnh"])

            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    asp.sort_values("mentions"),
                    x="mentions",
                    y="Khía cạnh",
                    orientation="h",
                    color="negative_rate",
                    color_continuous_scale="RdYlGn_r",
                    labels={
                        "mentions": "Lượt đề cập",
                        "negative_rate": "Tỷ lệ tiêu cực",
                    }
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                fig = px.scatter(
                    asp,
                    x="mentions",
                    y="negative_rate",
                    size="helpful_votes",
                    hover_name="Khía cạnh",
                    color="verified_rate",
                    labels={
                        "mentions": "Lượt đề cập",
                        "negative_rate": "Tỷ lệ tiêu cực",
                        "verified_rate": "Verified Purchase",
                        "helpful_votes": "Helpful votes",
                    }
                )
                fig.update_yaxes(tickformat=".0%")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if emerging_full.empty:
            st.info("Chưa có dữ liệu tín hiệu bất thường.")
        else:
            em = emerging_full.copy()

            if selected_asins:
                em = em[em["parent_asin"].astype(str).isin(selected_asins)]

            em["Khía cạnh"] = em["aspect"].map(ASPECT_VI).fillna(em["aspect"])
            em["Mức tín hiệu"] = em["signal_level"].map(SIGNAL_VI).fillna(em["signal_level"])

            top = em.sort_values("signal_score", ascending=False).head(30)

            cols = [
                c for c in [
                    "product_title",
                    "store",
                    "Khía cạnh",
                    "review_month",
                    "complaints",
                    "complaint_rate",
                    "growth_pct",
                    "signal_score",
                    "Mức tín hiệu",
                ]
                if c in top.columns
            ]

            show = top[cols].rename(columns={
                "product_title": "Sản phẩm",
                "store": "Cửa hàng / thương hiệu",
                "review_month": "Tháng",
                "complaints": "Số khiếu nại",
                "complaint_rate": "Tỷ lệ khiếu nại",
                "growth_pct": "Mức tăng",
                "signal_score": "Điểm tín hiệu",
            })

            st.dataframe(show, use_container_width=True, hide_index=True)

    with tab3:
        sub1, sub2 = st.tabs(["Sai lệch sao & nội dung", "Lời hứa & trải nghiệm"])

        with sub1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Review kiểm tra", fmt_int(nlp_validation.get("sample_rows")))
            c2.metric("Trường hợp lệch", fmt_int(nlp_validation.get("mismatch_rows")))
            c3.metric("Tỷ lệ lệch", fmt_pct(nlp_validation.get("mismatch_rate")))

            st.caption(
                f"Mô hình NLP: {nlp_validation.get('model', 'Không rõ')} · "
                f"Thiết bị: {nlp_validation.get('device', 'Không rõ')}"
            )

            if not nlp_sample.empty and "rating_text_mismatch" in nlp_sample.columns:
                mm = nlp_sample[nlp_sample["rating_text_mismatch"] == True].copy()

                cols = [
                    c for c in [
                        "product_title",
                        "rating",
                        "rating_sentiment",
                        "text_sentiment",
                        "text_sentiment_score",
                        "text",
                    ]
                    if c in mm.columns
                ]

                show = mm[cols].rename(columns={
                    "product_title": "Sản phẩm",
                    "rating": "Số sao",
                    "rating_sentiment": "Cảm xúc theo rating",
                    "text_sentiment": "Cảm xúc theo văn bản",
                    "text_sentiment_score": "Độ tin cậy NLP",
                    "text": "Nội dung review",
                })

                st.dataframe(show.head(100), use_container_width=True, hide_index=True)

        with sub2:
            if claim_full.empty:
                st.info("Chưa có dữ liệu đối chiếu lời hứa và trải nghiệm.")
            else:
                cl = claim_full.copy()
                if selected_asins:
                    cl = cl[cl["parent_asin"].astype(str).isin(selected_asins)]

                cl["Khía cạnh"] = cl["aspect"].map(ASPECT_VI).fillna(cl["aspect"])
                cl["Trạng thái"] = cl["experience_label"].map(EXPERIENCE_VI).fillna(cl["experience_label"])

                cols = [
                    c for c in [
                        "product_title",
                        "store",
                        "Khía cạnh",
                        "mentions",
                        "positive_rate",
                        "negative_rate",
                        "Trạng thái",
                    ]
                    if c in cl.columns
                ]

                show = cl[cols].rename(columns={
                    "product_title": "Sản phẩm",
                    "store": "Cửa hàng / thương hiệu",
                    "mentions": "Lượt đề cập",
                    "positive_rate": "Tỷ lệ tích cực",
                    "negative_rate": "Tỷ lệ tiêu cực",
                })

                st.dataframe(
                    show.sort_values(["Tỷ lệ tiêu cực", "Lượt đề cập"], ascending=[False, False]).head(100),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# PAGE 5 - BIG DATA & PHƯƠNG PHÁP
# ============================================================

elif page == "5. Big Data & phương pháp":

    hero(
        "Big Data & phương pháp",
        "Minh chứng quy trình xử lý dữ liệu lớn bằng Apache Spark, chất lượng dữ liệu và bước NLP phục vụ dashboard.",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Số dòng phân tích", fmt_int(pipeline.get("analysis_reviews")))
    c2.metric("Tỷ lệ ghép metadata", fmt_pct(pipeline.get("metadata_match_rate")))
    c3.metric("Sự kiện khía cạnh", fmt_int(pipeline.get("aspect_events")))
    c4.metric("Sản phẩm xuất web", fmt_int(pipeline.get("web_products")))

    st.markdown(
        f"""
        <div class="insight">
            <b>Quy trình:</b>
            Google Drive → Apache Spark → Làm sạch → Ghép metadata →
            Phân tích sản phẩm → Khai phá khía cạnh → Phát hiện vấn đề →
            NLP → Xuất Parquet/JSON → Streamlit.
            <br><br>
            <b>Phạm vi phân tích:</b>
            {pipeline.get('method_note', overview.get('analysis_scope_note', ''))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Kiến trúc xử lý")

    st.code(
        """
Amazon Reviews + Metadata
        ↓
Google Drive
        ↓
Apache Spark
        ↓
ETL + Làm sạch + Join Metadata
        ↓
Phân tích Review / Sản phẩm / Danh mục
        ↓
Aspect Mining
        ↓
Complaint Mining
        ↓
Emerging Issue Detection
        ↓
Risk Prioritization
        ↓
Transformer NLP
        ↓
Web Bundle
        ↓
Streamlit Dashboard
        """,
        language="text",
    )

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Benchmark Apache Spark")

        if not spark_benchmark.empty:
            fig = px.bar(
                spark_benchmark,
                x="target_rows",
                y="seconds",
                text="seconds",
                labels={
                    "target_rows": "Số dòng xử lý",
                    "seconds": "Thời gian (giây)",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.3f}s",
                textposition="outside"
            )

            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Thời gian benchmark có thể chịu ảnh hưởng bởi JVM warm-up, cache "
                "và Adaptive Query Execution của Spark."
            )

    with right:
        st.markdown("### Chất lượng dữ liệu")

        if not data_quality.empty:
            st.dataframe(
                data_quality,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### Cách hiểu điểm ưu tiên sản phẩm")

    st.info(
        "Điểm ưu tiên 0–100 được dùng để xếp thứ tự kiểm tra sản phẩm. "
        "Trong pipeline hiện tại, trọng số gồm: 35% tỷ lệ review tiêu cực, "
        "35% tín hiệu phản hồi tăng bất thường, 20% thành phần rating thấp "
        "và 10% helpful votes."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Ứng dụng Apache Spark và NLP trong khai phá tiếng nói khách hàng
        đối với sản phẩm Baby Products trên Amazon
    </div>
    """,
    unsafe_allow_html=True,
)
