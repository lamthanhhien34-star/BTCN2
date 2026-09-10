
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Phân tích đánh giá sản phẩm Baby trên Amazon",
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
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        padding: 24px 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        box-shadow: 0 16px 40px rgba(79, 70, 229, 0.18);
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
    }

    .hero p {
        margin: 8px 0 0 0;
        opacity: 0.92;
        font-size: 1rem;
    }

    .kpi {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        min-height: 112px;
    }

    .kpi-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .kpi-value {
        color: #111827;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 6px;
    }

    .kpi-note {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    .insight {
        background: white;
        border-left: 5px solid #6366f1;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 12px 0 18px 0;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #111827;
        margin: 10px 0 8px 0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 24px 0 12px 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 12px 14px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. ĐƯỜNG DẪN DỮ LIỆU
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

if not DATA_DIR.exists():
    DATA_DIR = ROOT


# ============================================================
# 4. HÀM ĐỌC DỮ LIỆU
# ============================================================

@st.cache_data(show_spinner=False)
def doc_json(ten_file):
    duong_dan = DATA_DIR / ten_file
    if not duong_dan.exists():
        return {}
    with open(duong_dan, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def doc_csv(ten_file):
    duong_dan = DATA_DIR / ten_file
    if not duong_dan.exists():
        return pd.DataFrame()
    return pd.read_csv(duong_dan)


@st.cache_data(show_spinner=False)
def doc_parquet(ten_file):
    duong_dan = DATA_DIR / ten_file
    if not duong_dan.exists():
        return pd.DataFrame()
    return pd.read_parquet(duong_dan)


def doi_ngay(df, cot):
    if not df.empty and cot in df.columns:
        df = df.copy()
        df[cot] = pd.to_datetime(df[cot], errors="coerce")
    return df


def dinh_dang_so(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(float(x))):,}"


def dinh_dang_ty_le(x, so_le=1):
    if pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{so_le}f}%"


def dinh_dang_tien(x):
    if pd.isna(x):
        return "—"
    return f"${float(x):,.2f}"


def hien_kpi(nhan, gia_tri, ghi_chu=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{nhan}</div>
            <div class="kpi-value">{gia_tri}</div>
            <div class="kpi-note">{ghi_chu}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hien_tieu_de(tieu_de, mo_ta):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{tieu_de}</h1>
            <p>{mo_ta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def thong_bao_khong_co_du_lieu(noi_dung="Không có dữ liệu phù hợp với bộ lọc hiện tại."):
    st.info(noi_dung)


# ============================================================
# 5. ĐỌC TOÀN BỘ FILE KẾT QUẢ
# ============================================================

tong_quan = doc_json("overview.json")
chi_so_pipeline = doc_json("pipeline_metrics.json")
kiem_dinh_nlp = doc_json("nlp_validation.json")

san_pham = doc_parquet("product_summary.parquet")
xu_huong_thang = doi_ngay(doc_parquet("monthly_trend.parquet"), "review_month")
tom_tat_danh_muc = doc_parquet("subcategory_summary.parquet")
xu_huong_san_pham = doi_ngay(doc_parquet("product_monthly.parquet"), "review_month")
tom_tat_khia_canh = doc_parquet("aspect_summary.parquet")
khia_canh_san_pham = doc_parquet("aspect_product_summary.parquet")
danh_gia_dai_dien = doi_ngay(doc_parquet("representative_reviews.parquet"), "review_date")
khieu_nai_theo_thang = doi_ngay(doc_parquet("complaint_monthly.parquet"), "review_month")
tin_hieu_bat_thuong = doi_ngay(doc_parquet("emerging_issues.parquet"), "review_month")
diem_rui_ro = doc_parquet("risk_scores.parquet")
loi_hua_trai_nghiem = doc_parquet("claim_experience.parquet")
mau_nlp = doi_ngay(doc_parquet("nlp_sentiment_sample.parquet"), "review_date")
benchmark = doc_csv("spark_benchmark.csv")
chat_luong_du_lieu = doc_csv("data_quality.csv")


# ============================================================
# 6. THANH BÊN
# ============================================================

with st.sidebar:
    st.markdown("## 🧸 PHÂN TÍCH BABY PRODUCTS")
    st.caption("Amazon Review Intelligence")

    trang = st.radio(
        "Chọn nội dung",
        [
            "1. Tổng quan",
            "2. Khám phá sản phẩm",
            "3. Tiếng nói khách hàng",
            "4. Cảnh báo & rủi ro",
            "5. So sánh sản phẩm",
            "6. Phân tích NLP",
            "7. Phòng thí nghiệm Big Data",
        ],
    )

    st.markdown("---")
    st.markdown("### Bộ lọc")

    danh_muc_co_san = []
    if not san_pham.empty and "subcategory" in san_pham.columns:
        danh_muc_co_san = sorted(
            san_pham["subcategory"].dropna().astype(str).unique().tolist()
        )

    danh_muc_chon = st.multiselect(
        "Danh mục con",
        danh_muc_co_san,
        placeholder="Tất cả danh mục",
    )

    review_toi_thieu = st.slider(
        "Số review tối thiểu của sản phẩm",
        min_value=0,
        max_value=500,
        value=0,
        step=10,
    )

    st.markdown("---")

    che_do = tong_quan.get(
        "analysis_mode",
        chi_so_pipeline.get("analysis_mode", "FAST")
    )

    st.caption(f"Chế độ phân tích: {che_do}")
    st.caption(
        f"Số dòng phân tích: "
        f"{dinh_dang_so(chi_so_pipeline.get('analysis_reviews', tong_quan.get('reviews', 0)))}"
    )


# ============================================================
# 7. LỌC SẢN PHẨM
# ============================================================

san_pham_loc = san_pham.copy()

if not san_pham_loc.empty:
    if danh_muc_chon and "subcategory" in san_pham_loc.columns:
        san_pham_loc = san_pham_loc[
            san_pham_loc["subcategory"].isin(danh_muc_chon)
        ]

    if "reviews" in san_pham_loc.columns:
        san_pham_loc = san_pham_loc[
            san_pham_loc["reviews"] >= review_toi_thieu
        ]

tap_asin = set()

if not san_pham_loc.empty and "parent_asin" in san_pham_loc.columns:
    tap_asin = set(
        san_pham_loc["parent_asin"].astype(str)
    )


# ============================================================
# TRANG 1 - TỔNG QUAN
# ============================================================

if trang == "1. Tổng quan":

    hien_tieu_de(
        "Tổng quan hoạt động đánh giá",
        "Bức tranh nhanh về quy mô dữ liệu, mức độ hài lòng và các tín hiệu đáng chú ý trong nhóm sản phẩm Baby Products.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        hien_kpi(
            "Số review",
            dinh_dang_so(tong_quan.get("reviews")),
            "Review sau làm sạch"
        )

    with c2:
        hien_kpi(
            "Số sản phẩm",
            dinh_dang_so(tong_quan.get("products")),
            "Sản phẩm được phân tích"
        )

    with c3:
        hien_kpi(
            "Điểm trung bình",
            f"{float(tong_quan.get('avg_rating', 0)):.2f} ★",
            "Thang điểm 1–5"
        )

    with c4:
        hien_kpi(
            "Review tiêu cực",
            dinh_dang_ty_le(tong_quan.get("negative_rate")),
            "Review 1–2 sao"
        )

    with c5:
        hien_kpi(
            "Mua hàng xác minh",
            dinh_dang_ty_le(tong_quan.get("verified_rate")),
            "Verified Purchase"
        )

    st.markdown(
        f"""
        <div class="insight">
            <b>Nhận xét nhanh:</b>
            Hệ thống đang phân tích <b>{dinh_dang_so(tong_quan.get('reviews'))}</b> review.
            Điểm đánh giá trung bình đạt <b>{float(tong_quan.get('avg_rating', 0)):.2f}/5</b>,
            trong khi tỷ lệ review tiêu cực là <b>{dinh_dang_ty_le(tong_quan.get('negative_rate'))}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    trai, phai = st.columns([1.5, 1])

    with trai:
        st.markdown('<div class="section-title">Xu hướng review theo thời gian</div>', unsafe_allow_html=True)

        if not xu_huong_thang.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=xu_huong_thang["review_month"],
                    y=xu_huong_thang["reviews"],
                    name="Số review",
                    marker_color="#6366f1",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=xu_huong_thang["review_month"],
                    y=xu_huong_thang["avg_rating"],
                    name="Điểm trung bình",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#f59e0b", width=3),
                )
            )

            fig.update_layout(
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(title="Số review"),
                yaxis2=dict(
                    title="Điểm trung bình",
                    overlaying="y",
                    side="right",
                    range=[1, 5],
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            thong_bao_khong_co_du_lieu()

    with phai:
        st.markdown('<div class="section-title">Sức khỏe danh mục</div>', unsafe_allow_html=True)

        if not tom_tat_danh_muc.empty:

            temp = tom_tat_danh_muc.copy()

            if danh_muc_chon:
                temp = temp[
                    temp["subcategory"].isin(danh_muc_chon)
                ]

            temp = temp.sort_values(
                "reviews",
                ascending=False
            ).head(12)

            fig = px.scatter(
                temp,
                x="avg_rating",
                y="negative_rate",
                size="reviews",
                hover_name="subcategory",
                color="reviews",
                labels={
                    "avg_rating": "Điểm trung bình",
                    "negative_rate": "Tỷ lệ tiêu cực",
                    "reviews": "Số review",
                },
            )

            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=400,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            thong_bao_khong_co_du_lieu()

    st.markdown('<div class="section-title">Tín hiệu bất thường mới nhất</div>', unsafe_allow_html=True)

    if not tin_hieu_bat_thuong.empty:

        temp = tin_hieu_bat_thuong.copy()

        if tap_asin:
            temp = temp[
                temp["parent_asin"].astype(str).isin(tap_asin)
            ]

        temp = temp.sort_values(
            "signal_score",
            ascending=False
        ).head(15)

        cot_hien = [
            c for c in [
                "parent_asin",
                "aspect",
                "complaints",
                "complaint_rate",
                "growth_pct",
                "z_score",
                "signal_level",
            ]
            if c in temp.columns
        ]

        st.dataframe(
            temp[cot_hien],
            use_container_width=True,
            hide_index=True,
        )

    else:
        thong_bao_khong_co_du_lieu("Chưa có tín hiệu bất thường phù hợp.")


# ============================================================
# TRANG 2 - KHÁM PHÁ SẢN PHẨM
# ============================================================

elif trang == "2. Khám phá sản phẩm":

    hien_tieu_de(
        "Khám phá sản phẩm",
        "Chọn một sản phẩm để xem hiệu suất đánh giá, xu hướng theo thời gian, các khía cạnh khách hàng quan tâm và review tiêu biểu.",
    )

    if san_pham_loc.empty:
        thong_bao_khong_co_du_lieu("Không có sản phẩm phù hợp với bộ lọc.")

    else:

        tu_khoa = st.text_input(
            "🔎 Tìm theo tên sản phẩm hoặc Parent ASIN",
            placeholder="Ví dụ: bottle, stroller, diaper..."
        )

        danh_sach = san_pham_loc.copy()

        if tu_khoa.strip():

            q = tu_khoa.strip().lower()

            mask = (
                danh_sach["product_title"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(q, regex=False)
                |
                danh_sach["parent_asin"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(q, regex=False)
            )

            danh_sach = danh_sach[mask]

        danh_sach = danh_sach.sort_values(
            "reviews",
            ascending=False
        ).head(300)

        if danh_sach.empty:
            thong_bao_khong_co_du_lieu("Không tìm thấy sản phẩm.")

        else:

            danh_sach["ten_hien_thi"] = (
                danh_sach["product_title"]
                .fillna("Không rõ tên")
                .astype(str)
                .str.slice(0, 85)
                + " · "
                + danh_sach["parent_asin"].astype(str)
            )

            lua_chon = st.selectbox(
                "Chọn sản phẩm",
                danh_sach["ten_hien_thi"].tolist()
            )

            dong = danh_sach[
                danh_sach["ten_hien_thi"] == lua_chon
            ].iloc[0]

            asin = str(dong["parent_asin"])

            st.markdown(f"### {dong.get('product_title', 'Sản phẩm')}")
            st.caption(
                f"Parent ASIN: {asin} · "
                f"Danh mục: {dong.get('subcategory', 'Không rõ')} · "
                f"Cửa hàng: {dong.get('store', 'Không rõ')}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Số review", dinh_dang_so(dong.get("reviews")))
            c2.metric("Điểm trung bình", f"{float(dong.get('avg_rating', 0)):.2f} ★")
            c3.metric("Tỷ lệ tiêu cực", dinh_dang_ty_le(dong.get("negative_rate")))
            c4.metric("Verified Purchase", dinh_dang_ty_le(dong.get("verified_rate")))
            c5.metric("Giá", dinh_dang_tien(dong.get("price")))

            trai, phai = st.columns([1.45, 1])

            with trai:
                st.markdown('<div class="section-title">Xu hướng sản phẩm theo tháng</div>', unsafe_allow_html=True)

                temp = (
                    xu_huong_san_pham[
                        xu_huong_san_pham["parent_asin"].astype(str) == asin
                    ]
                    .sort_values("review_month")
                    if not xu_huong_san_pham.empty
                    else pd.DataFrame()
                )

                if not temp.empty:

                    fig = go.Figure()

                    fig.add_trace(
                        go.Bar(
                            x=temp["review_month"],
                            y=temp["reviews"],
                            name="Số review",
                            marker_color="#6366f1",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=temp["review_month"],
                            y=temp["negative_rate"],
                            name="Tỷ lệ tiêu cực",
                            yaxis="y2",
                            mode="lines+markers",
                            line=dict(color="#ef4444", width=3),
                        )
                    )

                    fig.update_layout(
                        height=380,
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
                    thong_bao_khong_co_du_lieu(
                        "Sản phẩm này chưa có dữ liệu xu hướng theo tháng."
                    )

            with phai:
                st.markdown('<div class="section-title">Khía cạnh khách hàng quan tâm</div>', unsafe_allow_html=True)

                temp = (
                    khia_canh_san_pham[
                        khia_canh_san_pham["parent_asin"].astype(str) == asin
                    ].copy()
                    if not khia_canh_san_pham.empty
                    else pd.DataFrame()
                )

                if not temp.empty:

                    temp = temp.sort_values(
                        "mentions",
                        ascending=True
                    )

                    fig = px.bar(
                        temp,
                        x="negative_rate",
                        y="aspect",
                        orientation="h",
                        color="mentions",
                        labels={
                            "negative_rate": "Tỷ lệ tiêu cực",
                            "aspect": "",
                            "mentions": "Lượt đề cập",
                        },
                    )

                    fig.update_xaxes(tickformat=".0%")
                    fig.update_layout(
                        height=380,
                        coloraxis_showscale=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    thong_bao_khong_co_du_lieu(
                        "Chưa có dữ liệu khía cạnh cho sản phẩm này."
                    )

            st.markdown('<div class="section-title">Review tiêu biểu</div>', unsafe_allow_html=True)

            temp = (
                danh_gia_dai_dien[
                    danh_gia_dai_dien["parent_asin"].astype(str) == asin
                ].copy()
                if not danh_gia_dai_dien.empty
                else pd.DataFrame()
            )

            if not temp.empty:

                cam_xuc = st.selectbox(
                    "Lọc theo cảm xúc",
                    ["Tất cả", "Negative", "Neutral", "Positive"]
                )

                if cam_xuc != "Tất cả":
                    temp = temp[
                        temp["rating_sentiment"] == cam_xuc
                    ]

                for _, r in temp.head(12).iterrows():

                    with st.expander(
                        f"{int(round(r.get('rating', 0)))}★ · "
                        f"{r.get('rating_sentiment', '')} · "
                        f"Helpful: {dinh_dang_so(r.get('helpful_vote', 0))}"
                    ):
                        st.write(r.get("text", ""))
                        st.caption(
                            "Mua hàng xác minh: "
                            + ("Có" if bool(r.get("verified_purchase", False)) else "Không")
                        )

            else:
                thong_bao_khong_co_du_lieu(
                    "Không có review tiêu biểu cho sản phẩm này."
                )


# ============================================================
# TRANG 3 - TIẾNG NÓI KHÁCH HÀNG
# ============================================================

elif trang == "3. Tiếng nói khách hàng":

    hien_tieu_de(
        "Tiếng nói khách hàng",
        "Phân tích những khía cạnh được nhắc đến nhiều nhất và xác định các chủ đề có tỷ lệ phản hồi tiêu cực cao.",
    )

    if tom_tat_khia_canh.empty:
        thong_bao_khong_co_du_lieu()

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Số khía cạnh",
            dinh_dang_so(
                tom_tat_khia_canh["aspect"].nunique()
            )
        )

        c2.metric(
            "Tổng lượt đề cập",
            dinh_dang_so(
                tom_tat_khia_canh["mentions"].sum()
            )
        )

        khia_canh_noi_bat = (
            tom_tat_khia_canh
            .sort_values("mentions", ascending=False)
            .iloc[0]["aspect"]
        )

        c3.metric(
            "Được nhắc nhiều nhất",
            str(khia_canh_noi_bat)
        )

        trai, phai = st.columns([1.2, 1])

        with trai:

            temp = tom_tat_khia_canh.sort_values(
                "mentions",
                ascending=True
            )

            fig = px.bar(
                temp,
                x="mentions",
                y="aspect",
                orientation="h",
                color="negative_rate",
                color_continuous_scale="RdYlGn_r",
                labels={
                    "mentions": "Lượt đề cập",
                    "aspect": "",
                    "negative_rate": "Tỷ lệ tiêu cực",
                },
            )

            fig.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=20, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

        with phai:

            fig = px.scatter(
                tom_tat_khia_canh,
                x="mentions",
                y="negative_rate",
                size="helpful_votes",
                hover_name="aspect",
                color="verified_rate",
                labels={
                    "mentions": "Lượt đề cập",
                    "negative_rate": "Tỷ lệ tiêu cực",
                    "helpful_votes": "Helpful votes",
                    "verified_rate": "Verified Purchase",
                },
            )

            fig.update_yaxes(tickformat=".0%")

            fig.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=20, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

        khia_canh_chon = st.selectbox(
            "Chọn khía cạnh để xem sâu hơn",
            tom_tat_khia_canh
            .sort_values("mentions", ascending=False)["aspect"]
            .tolist()
        )

        temp = (
            khia_canh_san_pham[
                khia_canh_san_pham["aspect"] == khia_canh_chon
            ].copy()
            if not khia_canh_san_pham.empty
            else pd.DataFrame()
        )

        if tap_asin and not temp.empty:
            temp = temp[
                temp["parent_asin"].astype(str).isin(tap_asin)
            ]

        if not temp.empty:

            temp = temp.sort_values(
                ["negative_rate", "mentions"],
                ascending=[False, False],
            ).head(20)

            st.markdown(
                f"### Sản phẩm có tỷ lệ phản hồi tiêu cực cao về: {khia_canh_chon}"
            )

            st.dataframe(
                temp,
                use_container_width=True,
                hide_index=True,
            )

        else:
            thong_bao_khong_co_du_lieu()


# ============================================================
# TRANG 4 - CẢNH BÁO & RỦI RO
# ============================================================

elif trang == "4. Cảnh báo & rủi ro":

    hien_tieu_de(
        "Cảnh báo & rủi ro",
        "Ưu tiên các sản phẩm cần theo dõi dựa trên review tiêu cực, xu hướng khiếu nại và mức độ bất thường.",
    )

    st.warning(
        "Điểm rủi ro chỉ là chỉ số ưu tiên phân tích dựa trên review. "
        "Không phải chứng nhận an toàn hay kết luận chất lượng sản phẩm."
    )

    if diem_rui_ro.empty:

        thong_bao_khong_co_du_lieu()

    else:

        temp = diem_rui_ro.copy()

        if danh_muc_chon and "subcategory" in temp.columns:
            temp = temp[
                temp["subcategory"].isin(danh_muc_chon)
            ]

        if "reviews" in temp.columns:
            temp = temp[
                temp["reviews"] >= review_toi_thieu
            ]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Sản phẩm được chấm điểm",
            dinh_dang_so(len(temp))
        )

        c2.metric(
            "Rủi ro rất cao",
            dinh_dang_so(
                (temp["risk_level"] == "Critical").sum()
            )
        )

        c3.metric(
            "Rủi ro cao",
            dinh_dang_so(
                (temp["risk_level"] == "High").sum()
            )
        )

        c4.metric(
            "Điểm rủi ro trung vị",
            f"{temp['risk_score'].median():.1f}"
        )

        top = temp.sort_values(
            "risk_score",
            ascending=False
        ).head(20)

        fig = px.bar(
            top.sort_values("risk_score"),
            x="risk_score",
            y="product_title",
            orientation="h",
            color="risk_level",
            color_discrete_map={
                "Critical": "#dc2626",
                "High": "#f97316",
                "Medium": "#eab308",
                "Low": "#16a34a",
            },
            labels={
                "risk_score": "Điểm rủi ro",
                "product_title": "",
                "risk_level": "Mức độ",
            },
        )

        fig.update_layout(
            height=680,
            margin=dict(l=10, r=10, t=20, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tín hiệu bất thường theo khía cạnh")

        if not tin_hieu_bat_thuong.empty:

            em = tin_hieu_bat_thuong.copy()

            if tap_asin:
                em = em[
                    em["parent_asin"].astype(str).isin(tap_asin)
                ]

            em = em.sort_values(
                "signal_score",
                ascending=False
            ).head(30)

            st.dataframe(
                em,
                use_container_width=True,
                hide_index=True,
            )

        else:
            thong_bao_khong_co_du_lieu()


# ============================================================
# TRANG 5 - SO SÁNH SẢN PHẨM
# ============================================================

elif trang == "5. So sánh sản phẩm":

    hien_tieu_de(
        "So sánh sản phẩm",
        "So sánh trực tiếp nhiều sản phẩm trên rating, tỷ lệ tiêu cực, verified purchase và các khía cạnh trải nghiệm.",
    )

    if san_pham_loc.empty:
        thong_bao_khong_co_du_lieu()

    else:

        temp = san_pham_loc.sort_values(
            "reviews",
            ascending=False
        ).head(500).copy()

        temp["ten_hien_thi"] = (
            temp["product_title"]
            .fillna("Không rõ tên")
            .astype(str)
            .str.slice(0, 70)
            + " · "
            + temp["parent_asin"].astype(str)
        )

        mac_dinh = temp["ten_hien_thi"].head(
            min(3, len(temp))
        ).tolist()

        chon = st.multiselect(
            "Chọn từ 2 đến 4 sản phẩm",
            temp["ten_hien_thi"].tolist(),
            default=mac_dinh,
            max_selections=4,
        )

        so_sanh = temp[
            temp["ten_hien_thi"].isin(chon)
        ].copy()

        if len(so_sanh) < 2:

            st.info("Hãy chọn ít nhất 2 sản phẩm để so sánh.")

        else:

            cot = st.columns(len(so_sanh))

            for c, (_, r) in zip(cot, so_sanh.iterrows()):
                with c:
                    st.markdown(
                        f"**{str(r['product_title'])[:65]}**"
                    )
                    st.metric(
                        "Điểm trung bình",
                        f"{float(r['avg_rating']):.2f} ★"
                    )
                    st.caption(
                        f"{dinh_dang_so(r['reviews'])} review · "
                        f"Tiêu cực {dinh_dang_ty_le(r['negative_rate'])}"
                    )

            melt = so_sanh[
                [
                    "product_title",
                    "positive_rate",
                    "negative_rate",
                    "verified_rate",
                ]
            ].melt(
                id_vars=["product_title"],
                var_name="Chỉ số",
                value_name="Tỷ lệ",
            )

            melt["Chỉ số"] = melt["Chỉ số"].replace({
                "positive_rate": "Tích cực",
                "negative_rate": "Tiêu cực",
                "verified_rate": "Verified Purchase",
            })

            fig = px.bar(
                melt,
                x="product_title",
                y="Tỷ lệ",
                color="Chỉ số",
                barmode="group",
                labels={
                    "product_title": "",
                },
            )

            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=430)

            st.plotly_chart(fig, use_container_width=True)

            asin_chon = so_sanh[
                "parent_asin"
            ].astype(str).tolist()

            khia_canh = (
                khia_canh_san_pham[
                    khia_canh_san_pham["parent_asin"]
                    .astype(str)
                    .isin(asin_chon)
                ].copy()
                if not khia_canh_san_pham.empty
                else pd.DataFrame()
            )

            if not khia_canh.empty:

                bang_ten = dict(
                    zip(
                        so_sanh["parent_asin"].astype(str),
                        so_sanh["product_title"].astype(str),
                    )
                )

                khia_canh["Sản phẩm"] = (
                    khia_canh["parent_asin"]
                    .astype(str)
                    .map(bang_ten)
                )

                fig = px.bar(
                    khia_canh,
                    x="aspect",
                    y="negative_rate",
                    color="Sản phẩm",
                    barmode="group",
                    labels={
                        "aspect": "Khía cạnh",
                        "negative_rate": "Tỷ lệ tiêu cực",
                    },
                )

                fig.update_yaxes(tickformat=".0%")
                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TRANG 6 - PHÂN TÍCH NLP
# ============================================================

elif trang == "6. Phân tích NLP":

    hien_tieu_de(
        "Phân tích NLP",
        "Tìm kiếm bằng chứng trong review, đối chiếu lời hứa sản phẩm với trải nghiệm thực tế và kiểm tra sự lệch giữa số sao và nội dung văn bản.",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🔎 Tìm trong review",
            "🎯 Lời hứa & trải nghiệm",
            "🧠 Sai lệch sao & nội dung",
        ]
    )

    with tab1:

        st.markdown("### Tìm kiếm review theo từ khóa")

        tu_khoa = st.text_input(
            "Nhập từ khóa",
            placeholder="Ví dụ: leak, smell, cleaning, easy to use..."
        )

        cam_xuc = st.selectbox(
            "Cảm xúc",
            ["Tất cả", "Negative", "Neutral", "Positive"]
        )

        temp = danh_gia_dai_dien.copy()

        if tap_asin and not temp.empty:
            temp = temp[
                temp["parent_asin"].astype(str).isin(tap_asin)
            ]

        if cam_xuc != "Tất cả" and not temp.empty:
            temp = temp[
                temp["rating_sentiment"] == cam_xuc
            ]

        if tu_khoa.strip() and not temp.empty:

            q = tu_khoa.strip().lower()

            mask = (
                temp["text"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(q, regex=False)
                |
                temp["title"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(q, regex=False)
            )

            temp = temp[mask]

        if tu_khoa.strip():

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Review phù hợp",
                dinh_dang_so(len(temp))
            )

            if not temp.empty:

                c2.metric(
                    "Tỷ lệ tiêu cực",
                    dinh_dang_ty_le(
                        (
                            temp["rating_sentiment"] == "Negative"
                        ).mean()
                    )
                )

                c3.metric(
                    "Điểm trung bình",
                    f"{temp['rating'].mean():.2f}"
                )

                for _, r in (
                    temp
                    .sort_values("helpful_vote", ascending=False)
                    .head(15)
                    .iterrows()
                ):

                    with st.expander(
                        f"{r.get('product_title', 'Sản phẩm')} · "
                        f"{r.get('rating', 0):.0f}★ · "
                        f"{r.get('rating_sentiment', '')}"
                    ):
                        st.write(r.get("text", ""))
                        st.caption(
                            f"Helpful: {dinh_dang_so(r.get('helpful_vote', 0))} · "
                            f"Verified: {'Có' if bool(r.get('verified_purchase', False)) else 'Không'}"
                        )

            else:
                thong_bao_khong_co_du_lieu(
                    "Không tìm thấy review phù hợp."
                )

    with tab2:

        if loi_hua_trai_nghiem.empty:

            thong_bao_khong_co_du_lieu()

        else:

            temp = loi_hua_trai_nghiem.copy()

            if tap_asin:
                temp = temp[
                    temp["parent_asin"].astype(str).isin(tap_asin)
                ]

            bang_nhan = {
                "Frequently contradicted": "Thường xuyên bị phản bác",
                "Mixed": "Trải nghiệm trái chiều",
                "Mostly confirmed": "Phần lớn được xác nhận",
                "Insufficient evidence": "Chưa đủ bằng chứng",
            }

            temp["Trạng thái"] = (
                temp["experience_label"]
                .map(bang_nhan)
                .fillna(temp["experience_label"])
            )

            dem = (
                temp["Trạng thái"]
                .value_counts()
                .reset_index()
            )

            dem.columns = [
                "Trạng thái",
                "Số trường hợp",
            ]

            fig = px.bar(
                dem,
                x="Trạng thái",
                y="Số trường hợp",
                color="Trạng thái",
            )

            fig.update_layout(
                height=360,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                temp.sort_values(
                    ["negative_rate", "mentions"],
                    ascending=[False, False],
                ).head(100),
                use_container_width=True,
                hide_index=True,
            )

    with tab3:

        if mau_nlp.empty:

            thong_bao_khong_co_du_lieu()

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Số review kiểm tra",
                dinh_dang_so(
                    kiem_dinh_nlp.get(
                        "sample_rows",
                        len(mau_nlp)
                    )
                )
            )

            c2.metric(
                "Số trường hợp lệch",
                dinh_dang_so(
                    kiem_dinh_nlp.get(
                        "mismatch_rows",
                        0
                    )
                )
            )

            c3.metric(
                "Tỷ lệ lệch",
                dinh_dang_ty_le(
                    kiem_dinh_nlp.get(
                        "mismatch_rate",
                        0
                    )
                )
            )

            st.caption(
                f"Mô hình: {kiem_dinh_nlp.get('model', 'Không rõ')} · "
                f"Thiết bị: {kiem_dinh_nlp.get('device', 'Không rõ')}"
            )

            if "rating_text_mismatch" in mau_nlp.columns:

                lech = mau_nlp[
                    mau_nlp["rating_text_mismatch"] == True
                ].copy()

                cot_hien = [
                    c for c in [
                        "product_title",
                        "rating",
                        "rating_sentiment",
                        "text_sentiment",
                        "text_sentiment_score",
                        "text",
                    ]
                    if c in lech.columns
                ]

                st.dataframe(
                    lech[cot_hien].head(100),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# TRANG 7 - BIG DATA LAB
# ============================================================

elif trang == "7. Phòng thí nghiệm Big Data":

    hien_tieu_de(
        "Phòng thí nghiệm Big Data",
        "Minh chứng kỹ thuật cho quy trình xử lý: quy mô dữ liệu, chất lượng dữ liệu, benchmark Apache Spark và kiến trúc triển khai.",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số dòng phân tích",
        dinh_dang_so(
            chi_so_pipeline.get("analysis_reviews")
        )
    )

    c2.metric(
        "Tỷ lệ ghép metadata",
        dinh_dang_ty_le(
            chi_so_pipeline.get("metadata_match_rate")
        )
    )

    c3.metric(
        "Số sự kiện khía cạnh",
        dinh_dang_so(
            chi_so_pipeline.get("aspect_events")
        )
    )

    c4.metric(
        "Số sản phẩm xuất web",
        dinh_dang_so(
            chi_so_pipeline.get("web_products")
        )
    )

    st.markdown(
        f"""
        <div class="insight">
            <b>Quy trình:</b>
            Google Drive → Apache Spark → Làm sạch → Ghép metadata →
            Phân tích sản phẩm → NLP → Cảnh báo → Parquet/JSON → Streamlit.
            <br><br>
            <b>Phạm vi:</b>
            {chi_so_pipeline.get('method_note', tong_quan.get('analysis_scope_note', ''))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    trai, phai = st.columns([1.2, 1])

    with trai:

        st.markdown('<div class="section-title">Benchmark Apache Spark</div>', unsafe_allow_html=True)

        if not benchmark.empty:

            fig = px.bar(
                benchmark,
                x="target_rows",
                y="seconds",
                text="seconds",
                labels={
                    "target_rows": "Số dòng xử lý",
                    "seconds": "Thời gian chạy (giây)",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.3f}s",
                textposition="outside"
            )

            fig.update_layout(
                height=390
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Lưu ý: thời gian benchmark có thể chịu ảnh hưởng bởi JVM warm-up, cache "
                "và Adaptive Query Execution của Spark."
            )

        else:
            thong_bao_khong_co_du_lieu()

    with phai:

        st.markdown('<div class="section-title">Chất lượng dữ liệu</div>', unsafe_allow_html=True)

        if not chat_luong_du_lieu.empty:

            st.dataframe(
                chat_luong_du_lieu,
                use_container_width=True,
                hide_index=True,
            )

        else:
            thong_bao_khong_co_du_lieu()

    st.markdown("### Kiến trúc xử lý")

    st.code(
        """
Google Drive
    ↓
Dữ liệu Amazon Baby Products
    ↓
Apache Spark
    ↓
Làm sạch + Chuẩn hóa + Ghép Metadata
    ↓
Phân tích sản phẩm
    ↓
Phân tích khía cạnh khách hàng
    ↓
Khai phá phản hồi tiêu cực
    ↓
Phát hiện tín hiệu bất thường
    ↓
Chấm điểm ưu tiên rủi ro
    ↓
NLP Transformer
    ↓
Web Bundle
    ↓
Streamlit Dashboard
        """,
        language="text"
    )


# ============================================================
# 8. CHÂN TRANG
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
