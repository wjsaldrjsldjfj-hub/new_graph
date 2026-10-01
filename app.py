import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_DATA_YEAR = 2025
MIN_DAYS = 300


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    return df


# --------------------------------
# 데이터 준비
# --------------------------------
df = load_data()

yearly = (
    df[df["연도"] <= LAST_DATA_YEAR]
    .groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count"),
    )
    .reset_index()
)

# 관측일수가 300일 미만인 연도 제외
yearly = yearly[yearly["관측일수"] >= MIN_DAYS].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# 1908년부터 지난 연수
yearly["경과연수"] = yearly["연도"] - BASE_YEAR


# --------------------------------
# 전체 기간 회귀
# --------------------------------
x_all = yearly["경과연수"].to_numpy()
y_all = yearly["평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)

# °C/년 → °C/100년
slope_all_100 = slope_all * 100

correlation = np.corrcoef(x_all, y_all)[0, 1]


# --------------------------------
# 최근 20년 회귀
# --------------------------------
recent_end_year = int(yearly["연도"].max())
recent_start_year = recent_end_year - 19

recent = yearly[
    (yearly["연도"] >= recent_start_year)
    & (yearly["연도"] <= recent_end_year)
].copy()

x_recent = recent["연도"].to_numpy()
y_recent = recent["평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1,
)

slope_recent_100 = slope_recent * 100


def predict_temperature(year):
    """전체 기간 회귀식으로 예상 평균기온 계산"""
    elapsed_years = year - BASE_YEAR
    return intercept_all + slope_all * elapsed_years


# --------------------------------
# 화면
# --------------------------------
st.title("🌡️ 서울 기온 예측기")

st.markdown(
    """
서울의 연평균기온 데이터를 이용해 선형회귀를 수행하고,
선택한 연도의 예상 평균기온을 확인합니다.
"""
)

# --------------------------------
# 회귀 기간 정보
# --------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{n_years}개",
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년",
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년",
    )

st.caption(
    f"2025년 이하이며 관측일수가 {MIN_DAYS}일 이상인 연도만 사용했습니다."
)


# --------------------------------
# 기울기 비교
# --------------------------------
st.subheader("📈 기온 상승률 비교")

st.markdown(
    "회귀 직선의 기울기를 **100년에 몇 °C 변하는가**로 환산했습니다."
)

slope_col1, slope_col2 = st.columns(2)

with slope_col1:
    st.metric(
        "전체 기간",
        f"{slope_all_100:+.2f} °C / 100년",
        help=f"{start_year}~{end_year}년 자료로 계산",
    )
    st.caption(
        f"{start_year}~{end_year}년, {n_years}개 연도"
    )

with slope_col2:
    st.metric(
        "최근 20년",
        f"{slope_recent_100:+.2f} °C / 100년",
        help=f"{recent_start_year}~{recent_end_year}년 자료로 계산",
    )
    st.caption(
        f"{recent_start_year}~{recent_end_year}년, "
        f"{len(recent)}개 연도"
    )


# --------------------------------
# 선택 연도
# --------------------------------
selected_year = st.slider(
    "예상 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

predicted_temp = predict_temperature(selected_year)

st.subheader(f"{selected_year}년 예상 평균기온")

st.markdown(
    f"""
    <div style="
        font-size: 3.5rem;
        font-weight: 700;
        color: #d62728;
        margin: 10px 0 25px 0;
    ">
        {predicted_temp:.2f} °C
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------
# 상관계수
# --------------------------------
st.subheader("상관계수")

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{correlation:.4f}",
)


# --------------------------------
# Plotly 그래프
# --------------------------------
line_years = np.arange(1900, 2101)

# 전체 기간 회귀선
line_temperatures_all = (
    intercept_all
    + slope_all * (line_years - BASE_YEAR)
)

# 최근 20년 회귀선
line_temperatures_recent = (
    intercept_recent
    + slope_recent * line_years
)

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#1f77b4",
            opacity=0.75,
        ),
        customdata=yearly["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures_all,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            color="#d62728",
            width=3,
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 기간 회귀: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures_recent,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            color="#2ca02c",
            width=3,
            dash="dash",
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "최근 20년 회귀: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 선택 연도 예상값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택 연도 예상값",
        marker=dict(
            size=14,
            color="#ff7f0e",
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            "예상 평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis=dict(
        title="연도",
        range=[1900, 2100],
        dtick=10,
        tickformat="d",
    ),
    yaxis=dict(
        title="평균기온 (°C)",
    ),
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    height=600,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# --------------------------------
# 회귀식
# --------------------------------
st.subheader("회귀식")

st.write("전체 기간 회귀식")

st.latex(
    rf"\hat{{T}} = {intercept_all:.4f} "
    rf"{slope_all:+.4f} \times (\mathrm{{연도}} - 1908)"
)

st.write(
    f"전체 기간 기울기: "
    f"**{slope_all_100:+.2f} °C / 100년**"
)

st.write("최근 20년 회귀식")

st.latex(
    rf"\hat{{T}} = {intercept_recent:.4f} "
    rf"{slope_recent:+.4f} \times \mathrm{{연도}}"
)

st.write(
    f"최근 20년 기울기: "
    f"**{slope_recent_100:+.2f} °C / 100년**"
)


# --------------------------------
# 데이터 보기
# --------------------------------
with st.expander("회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)

    st.dataframe(
        display_df[
            ["연도", "관측일수", "평균기온", "경과연수"]
        ],
        use_container_width=True,
        hide_index=True,
    )
